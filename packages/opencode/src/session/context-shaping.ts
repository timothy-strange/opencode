import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Token } from "@/util/token"
import { Identifier } from "@/id/id"
import { MessageID, PartID } from "./schema"

const OMITTED_MARKER = "[Earlier conversation omitted for local model context limit]"

export type SimpleInput = {
  readonly messages: SessionV1.WithParts[]
  readonly currentUserID: MessageID
  readonly budgetTokens: number
  readonly firstUserMaxChars: number
  readonly toolOutputMaxChars: number
  // How many of the most recent steps in the current turn keep full current-turn treatment
  // (larger tool-output preview, protected from omission). Older same-turn steps fall back to
  // age-based caps and become eligible for omission, so a long agentic loop does not accumulate
  // unbounded high-priority tool output.
  readonly currentTurnRecentSteps?: number
}

const DEFAULT_CURRENT_TURN_RECENT_STEPS = 3

export type Stats = {
  readonly inputMessages: number
  readonly outputMessages: number
  readonly omittedMessages: number
  readonly truncatedUserParts: number
  readonly truncatedToolOutputs: number
  readonly estimatedTokens: number
}

export type Result = {
  readonly messages: SessionV1.WithParts[]
  readonly stats: Stats
}

export type ToolScoreInput = {
  readonly tool: string
  readonly status: SessionV1.ToolPart["state"]["status"]
  readonly age: number
  readonly outputChars: number
  readonly currentTurn: boolean
  readonly compacted: boolean
  readonly hasFailureSignal: boolean
  readonly hasRecentDuplicate: boolean
  readonly inWorkingSet: boolean
}

export function forSimpleModel(input: SimpleInput): Result {
  const ordered = [...input.messages].sort((a, b) => compareID(a.info.id, b.info.id))
  const current = ordered.find((message) => message.info.id === input.currentUserID)
  if (!current || current.info.role !== "user") return emptyResult(input.messages)

  const first = ordered.find((message) => message.info.role === "user" && hasRealContent(message))
  const currentTurn = ordered.filter((message) => message.info.id >= current.info.id)
  const recentSteps = input.currentTurnRecentSteps ?? DEFAULT_CURRENT_TURN_RECENT_STEPS
  // Current user request plus the most recent steps of the turn keep full priority; the middle
  // steps of a long loop are demoted to omittable history.
  const recentTurn = new Set(currentTurn.slice(-Math.max(1, recentSteps)).map((message) => message.info.id))
  const protectedIds = new Set([...(first ? [first.info.id] : []), current.info.id, ...recentTurn])

  const tail = selectTail({
    messages: ordered.filter((message) => !protectedIds.has(message.info.id)),
    budgetTokens: input.budgetTokens,
  })
  const keep = new Set([...protectedIds, ...tail.messages.map((message) => message.info.id)])

  // Single chronological pass: keep protected/selected messages, cap the context copy of the
  // first user message, and collapse each run of dropped messages into one omission marker.
  const selected: SessionV1.WithParts[] = []
  let gap = false
  for (const message of ordered) {
    if (!keep.has(message.info.id)) {
      gap = true
      continue
    }
    if (gap) {
      selected.push(omissionMessage(current, message))
      gap = false
    }
    selected.push(
      first && message.info.id === first.info.id && first.info.id !== current.info.id
        ? capFirstUser(message, input.firstUserMaxChars)
        : message,
    )
  }

  const superseded = supersededToolParts(selected)
  const workingSet = workingSetPaths(current)
  const messages = dedupe(
    selected.map((message, index) =>
      shapeMessage({
        message,
        maxChars: input.toolOutputMaxChars,
        currentTurn: message.info.id === current.info.id || recentTurn.has(message.info.id),
        age: selected.length - index - 1,
        supersededToolParts: superseded,
        workingSet,
      }),
    ),
  )
  const stats = summarize({ input, messages, tailOmitted: tail.omitted })

  return { messages, stats }
}

function compareID(a: MessageID, b: MessageID) {
  if (a < b) return -1
  if (a > b) return 1
  return 0
}

export function scoreToolPart(input: ToolScoreInput) {
  if (input.compacted) return 0

  const status = (() => {
    if (input.status === "error") return 900
    if (input.status === "pending" || input.status === "running") return 650
    return 350
  })()
  const tool = (() => {
    if (input.tool === "read") return 260
    if (input.tool === "grep" || input.tool === "glob") return 220
    if (input.tool === "edit" || input.tool === "write") return 180
    if (input.tool === "webfetch") return 60
    if (input.tool === "bash") return -80
    return 0
  })()
  const sizePenalty = Math.min(300, Math.floor(input.outputChars / 1_000) * 25)
  const duplicatePenalty = input.hasRecentDuplicate ? 300 : 0

  return Math.max(
    0,
    status +
      tool +
      (input.currentTurn ? 500 : 0) +
      (input.hasFailureSignal ? 350 : 0) +
      (input.inWorkingSet ? 250 : 0) -
      input.age * 35 -
      sizePenalty -
      duplicatePenalty,
  )
}

function emptyResult(messages: SessionV1.WithParts[]): Result {
  return {
    messages,
    stats: {
      inputMessages: messages.length,
      outputMessages: messages.length,
      omittedMessages: 0,
      truncatedUserParts: 0,
      truncatedToolOutputs: 0,
      estimatedTokens: estimate(messages),
    },
  }
}

function hasRealContent(message: SessionV1.WithParts) {
  return !message.parts.every((part) => "synthetic" in part && part.synthetic)
}

function selectTail(input: { messages: SessionV1.WithParts[]; budgetTokens: number }) {
  const selected: SessionV1.WithParts[] = []
  let total = 0
  for (let i = input.messages.length - 1; i >= 0; i--) {
    const message = input.messages[i]
    if (!message) continue
    const next = estimate([message])
    if (selected.length > 0 && total + next > input.budgetTokens) break
    selected.unshift(message)
    total += next
    if (total >= input.budgetTokens) break
  }
  return {
    messages: selected,
    omitted: input.messages.length - selected.length,
  }
}

function capFirstUser(message: SessionV1.WithParts, maxChars: number) {
  if (maxChars <= 0) return message
  return {
    info: message.info,
    parts: message.parts.map((part) => {
      if (part.type !== "text") return part
      if (part.text.length <= maxChars) return part
      return {
        ...part,
        text: `${part.text.slice(0, maxChars)}\n[First user message truncated for local model context limit]`,
      }
    }),
  }
}

function shapeMessage(input: {
  message: SessionV1.WithParts
  maxChars: number
  currentTurn: boolean
  age: number
  supersededToolParts: Set<PartID>
  workingSet: Set<string>
}) {
  const isAssistant = input.message.info.role === "assistant"
  return {
    info: input.message.info,
    parts: input.message.parts.map((part) => {
      // Stale reads of a file that was read again later feed outdated content to weak
      // models, so drop them entirely regardless of size.
      if (part.type === "tool" && input.supersededToolParts.has(part.id)) return supersededReadMarker(part)
      // Long old assistant prose adds little for weak models; keep the current turn intact.
      if (part.type === "text" && isAssistant && !input.currentTurn) return capAssistantText(part, input.maxChars)
      if (part.type !== "tool") return part
      if (input.maxChars <= 0) return part
      if (part.state.status !== "completed") return part
      if (part.state.time.compacted) return part
      if (part.state.output.length <= input.maxChars) return part
      const hasFailureSignal = isFailureLikelyTool(part.tool) && containsFailureSignal(part.state.output)
      const score = scoreToolPart({
        tool: part.tool,
        status: part.state.status,
        age: input.age,
        outputChars: part.state.output.length,
        currentTurn: input.currentTurn,
        compacted: part.state.time.compacted !== undefined,
        hasFailureSignal,
        hasRecentDuplicate: input.supersededToolParts.has(part.id),
        inWorkingSet: toolReferencesWorkingSet(part, input.workingSet),
      })
      const maxChars = toolOutputLimit(input.maxChars, score)
      if (part.state.output.length <= maxChars) return part
      if (maxChars === 0) {
        return {
          ...part,
          state: {
            ...part.state,
            output: `[Tool output omitted for local model context limit: ${part.state.output.length} chars]`,
          },
        }
      }
      const omitted = part.state.output.length - maxChars
      return {
        ...part,
        state: {
          ...part.state,
          output: truncateToolOutput(part.state.output, maxChars, omitted, hasFailureSignal),
        },
      }
    }),
  }
}

function capAssistantText(part: SessionV1.TextPart, maxChars: number) {
  if (maxChars <= 0 || part.text.length <= maxChars) return part
  const omitted = part.text.length - maxChars
  return {
    ...part,
    text: `${part.text.slice(0, maxChars)}\n[Assistant message truncated for local model context limit: omitted ${omitted} chars]`,
  }
}

function supersededReadMarker(part: SessionV1.ToolPart) {
  const path = toolPath(part) ?? "file"
  const size = part.state.status === "completed" ? part.state.output.length : 0
  return {
    ...part,
    state: {
      ...part.state,
      output: `[Earlier read of ${path} superseded by a newer read for local model context limit: ${size} chars]`,
    },
  }
}

function supersededToolParts(messages: SessionV1.WithParts[]) {
  const latestReadByTarget = new Map<string, PartID>()
  const reads = messages.flatMap((message) =>
    message.parts.flatMap((part) => {
      const target = readTarget(part)
      return target ? [{ id: part.id, target }] : []
    }),
  )
  reads.forEach((item) => latestReadByTarget.set(item.target, item.id))
  return new Set(reads.flatMap((item) => (latestReadByTarget.get(item.target) === item.id ? [] : [item.id])))
}

function readTarget(part: SessionV1.Part) {
  if (part.type !== "tool") return
  if (part.tool !== "read") return
  if (part.state.status !== "completed") return
  return toolPath(part)
}

function toolPath(part: SessionV1.Part) {
  if (part.type !== "tool") return
  const input = "input" in part.state ? part.state.input : undefined
  if (!input || typeof input !== "object") return
  const value = (input as Record<string, unknown>).filePath ?? (input as Record<string, unknown>).path
  return typeof value === "string" ? value : undefined
}

function workingSetPaths(message: SessionV1.WithParts) {
  const text = message.parts
    .flatMap((part) => (part.type === "text" ? [part.text] : []))
    .join(" ")
  const matches = text.match(/[\w./@-]*[\w-]\.[a-zA-Z][\w]{0,7}\b/g) ?? []
  return new Set(matches.map((match) => match.toLowerCase()))
}

function toolReferencesWorkingSet(part: SessionV1.ToolPart, workingSet: Set<string>) {
  if (workingSet.size === 0) return false
  const path = toolPath(part)
  if (!path) return false
  const lower = path.toLowerCase()
  const base = lower.split("/").pop() ?? lower
  for (const entry of workingSet) {
    if (lower.endsWith(entry) || entry.endsWith(base)) return true
  }
  return false
}

function truncateToolOutput(output: string, maxChars: number, omitted: number, preferTail: boolean) {
  if (!preferTail)
    return `${output.slice(0, maxChars)}\n[Tool output truncated for local model context limit: omitted ${omitted} chars]`
  return `[Tool output truncated for local model context limit: omitted ${omitted} chars]\n${output.slice(-maxChars)}`
}

function isFailureLikelyTool(tool: string) {
  return tool === "bash"
}

function containsFailureSignal(output: string) {
  return /\b(error|failed|failure|exception|traceback|panic|timeout|denied|not found)\b/i.test(output)
}

function toolOutputLimit(maxChars: number, score: number) {
  if (score >= 800) return maxChars * 4
  if (score >= 500) return maxChars * 2
  if (score >= 250) return maxChars
  return 0
}

function omissionMessage(current: SessionV1.WithParts, anchor: SessionV1.WithParts) {
  const info = current.info as SessionV1.User
  const messageID = markerMessageID(anchor.info.id)
  return {
    info: {
      ...info,
      id: messageID,
      time: { created: info.time.created },
    },
    parts: [
      {
        id: PartID.ascending(),
        sessionID: info.sessionID,
        messageID,
        type: "text" as const,
        text: OMITTED_MARKER,
        synthetic: true,
      },
    ],
  }
}

// Derive an id that sorts just before the message the marker precedes, so the marker keeps
// its chronological position even if something downstream ever sorts by id.
function markerMessageID(anchorID: MessageID) {
  const before = Identifier.timestamp(anchorID) - 1
  if (!Number.isFinite(before) || before < 0) return MessageID.ascending()
  return MessageID.make(Identifier.create("msg", "ascending", before))
}

function dedupe(messages: SessionV1.WithParts[]) {
  const seen = new Set<MessageID>()
  return messages.filter((message) => {
    if (seen.has(message.info.id)) return false
    seen.add(message.info.id)
    return true
  })
}

function summarize(input: { input: SimpleInput; messages: SessionV1.WithParts[]; tailOmitted: number }): Stats {
  return {
    inputMessages: input.input.messages.length,
    outputMessages: input.messages.length,
    omittedMessages: input.tailOmitted,
    truncatedUserParts: countTruncatedUserParts(input.messages),
    truncatedToolOutputs: countTruncatedToolOutputs(input.input.messages, input.messages),
    estimatedTokens: estimate(input.messages),
  }
}

function countTruncatedUserParts(messages: SessionV1.WithParts[]) {
  return messages.reduce(
    (count, message) =>
      count +
      message.parts.filter(
        (part) => part.type === "text" && part.text.includes("[First user message truncated for local model context limit]"),
      ).length,
    0,
  )
}

function countTruncatedToolOutputs(original: SessionV1.WithParts[], shaped: SessionV1.WithParts[]) {
  const originalByPart = new Map(original.flatMap((message) => message.parts.map((part) => [part.id, part] as const)))
  return shaped.reduce(
    (count, message) =>
      count +
      message.parts.filter((part) => {
        const previous = originalByPart.get(part.id)
        return (
          part.type === "tool" &&
          previous?.type === "tool" &&
          part.state.status === "completed" &&
          previous.state.status === "completed" &&
          part.state.output !== previous.state.output
        )
      }).length,
    0,
  )
}

// Estimate the tokens the weak model actually sees (the flattened transcript), not the stored
// message envelope. JSON.stringify of SessionV1.WithParts is dominated by ids/metadata the model
// never receives, which would over-count small turns by an order of magnitude.
function renderMessage(message: SessionV1.WithParts) {
  return message.parts
    .flatMap((part) => {
      if (part.type === "text") return part.text ? [part.text] : []
      if (part.type !== "tool") return []
      const input = "input" in part.state ? JSON.stringify(part.state.input) : ""
      if (part.state.status === "completed") return [`${part.tool} result for ${input}:\n${part.state.output}`]
      if (part.state.status === "error") return [`${part.tool} error for ${input}: ${part.state.error ?? ""}`]
      return [`${part.tool} for ${input}`]
    })
    .join("\n")
}

function estimate(messages: SessionV1.WithParts[]) {
  return Token.estimate(messages.map(renderMessage).join("\n\n"))
}

export * as ContextShaping from "./context-shaping"
