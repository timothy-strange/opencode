import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Token } from "@/util/token"
import { MessageID, PartID } from "./schema"

const OMITTED_MARKER = "[Earlier conversation omitted for local model context limit]"

export type SimpleInput = {
  readonly messages: SessionV1.WithParts[]
  readonly currentUserID: MessageID
  readonly budgetTokens: number
  readonly firstUserMaxChars: number
  readonly toolOutputMaxChars: number
}

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
}

export function forSimpleModel(input: SimpleInput): Result {
  const current = input.messages.find((message) => message.info.id === input.currentUserID)
  if (!current || current.info.role !== "user") return emptyResult(input.messages)

  const first = input.messages.find((message) => message.info.role === "user" && hasRealContent(message))
  const currentTurn = input.messages.filter((message) => message.info.id >= current.info.id)
  const required = new Set([...currentTurn.map((message) => message.info.id), ...(first ? [first.info.id] : [])])
  const tail = selectTail({
    messages: input.messages.filter((message) => !required.has(message.info.id) && message.info.id < current.info.id),
    budgetTokens: input.budgetTokens,
  })
  const firstContext = first && first.info.id !== current.info.id ? [capFirstUser(first, input.firstUserMaxChars)] : []
  const selected = [
    ...firstContext,
    ...(tail.omitted > 0 ? [omissionMessage(current)] : []),
    ...tail.messages,
    ...currentTurn,
  ]
  const superseded = supersededToolParts(selected)
  const messages = dedupe(
    selected.map((message, index) =>
      capToolOutputs({
        message,
        maxChars: input.toolOutputMaxChars,
        currentTurn: message.info.id >= current.info.id,
        age: selected.length - index - 1,
        supersededToolParts: superseded,
      }),
    ),
  )
  const stats = summarize({ input, messages, tailOmitted: tail.omitted })

  return { messages, stats }
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
      (input.hasFailureSignal ? 350 : 0) -
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

function capToolOutputs(input: {
  message: SessionV1.WithParts
  maxChars: number
  currentTurn: boolean
  age: number
  supersededToolParts: Set<PartID>
}) {
  if (input.maxChars <= 0) return input.message
  return {
    info: input.message.info,
    parts: input.message.parts.map((part) => {
      if (part.type !== "tool") return part
      if (part.state.status !== "completed") return part
      if (part.state.time.compacted) return part
      if (part.state.output.length <= input.maxChars) return part
      const score = scoreToolPart({
        tool: part.tool,
        status: part.state.status,
        age: input.age,
        outputChars: part.state.output.length,
        currentTurn: input.currentTurn,
        compacted: part.state.time.compacted !== undefined,
        hasFailureSignal: containsFailureSignal(part.state.output),
        hasRecentDuplicate: input.supersededToolParts.has(part.id),
      })
      const maxChars = toolOutputLimit(input.maxChars, score)
      if (part.state.output.length <= maxChars) return part
      const preferTail = containsFailureSignal(part.state.output)
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
          output: truncateToolOutput(part.state.output, maxChars, omitted, preferTail),
        },
      }
    }),
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
  const input = part.state.input
  if (typeof input.filePath === "string") return input.filePath
  if (typeof input.path === "string") return input.path
}

function truncateToolOutput(output: string, maxChars: number, omitted: number, preferTail: boolean) {
  if (!preferTail)
    return `${output.slice(0, maxChars)}\n[Tool output truncated for local model context limit: omitted ${omitted} chars]`
  return `[Tool output truncated for local model context limit: omitted ${omitted} chars]\n${output.slice(-maxChars)}`
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

function omissionMessage(current: SessionV1.WithParts) {
  const info = current.info as SessionV1.User
  const messageID = MessageID.ascending()
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

function estimate(messages: SessionV1.WithParts[]) {
  return Token.estimate(JSON.stringify(messages))
}

export * as ContextShaping from "./context-shaping"
