import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import type { Auth } from "@/auth"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import type { RuntimeFlags } from "@/effect/runtime-flags"
import { InstanceState } from "@/effect/instance-state"
import { Permission } from "@/permission"
import type { Agent } from "@/agent/agent"
import type { MessageV2 } from "../message-v2"
import type { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import { SystemPrompt } from "../system"
import PROMPT_SIMPLE_COMPACTION from "../prompt/simple-compaction.txt"
import PROMPT_SIMPLE_EXPLORE from "../prompt/simple-explore.txt"
import PROMPT_SIMPLE_SUMMARY from "../prompt/simple-summary.txt"
import PROMPT_SIMPLE_TITLE from "../prompt/simple-title.txt"
import { InstallationVersion } from "@opencode-ai/core/installation/version"
import { Effect, Record } from "effect"
import { jsonSchema, tool as aiTool, type ModelMessage, type Tool } from "ai"
import type { Plugin } from "@/plugin"
import { mergeDeep } from "remeda"

const USER_AGENT = `opencode/${InstallationVersion}`

type PrepareInput = {
  readonly user: SessionV1.User
  readonly sessionID: string
  readonly parentSessionID?: string
  readonly model: Provider.Model
  readonly agent: Agent.Info
  readonly permission?: PermissionV1.Ruleset
  readonly system: string[]
  readonly messages: ModelMessage[]
  readonly small?: boolean
  readonly tools: Record<string, Tool>
  readonly provider: Provider.Info
  readonly auth: Auth.Info | undefined
  readonly plugin: Plugin.Interface
  readonly flags: RuntimeFlags.Info
  readonly isWorkflow: boolean
}

export type Prepared = {
  readonly system: string[]
  readonly messages: ModelMessage[]
  readonly tools: Record<string, Tool>
  readonly params: {
    readonly temperature?: number
    readonly topP?: number
    readonly topK?: number
    readonly maxOutputTokens?: number
    readonly options: Record<string, any>
  }
  readonly messageTransformOptions: Record<string, any>
  readonly headers: Record<string, string>
}

const mergeOptions = (target: Record<string, any>, source: Record<string, any> | undefined): Record<string, any> =>
  mergeDeep(target, source ?? {}) as Record<string, any>

export const prepare = Effect.fn("LLMRequestPrep.prepare")(function* (input: PrepareInput) {
  const isOpenaiOauth = input.provider.id === "openai" && input.auth?.type === "oauth"
  const simple = SystemPrompt.isSimple(input.model)
  const system = [
    [
      ...(simple
        ? simplePrompt(input.agent, input.model)
        : input.agent.prompt
          ? [input.agent.prompt]
          : SystemPrompt.provider(input.model)),
      ...input.system,
      ...(simple || !input.user.system ? [] : [input.user.system]),
    ]
      .filter((x) => x)
      .join("\n"),
  ]

  const header = system[0]
  if (!simple) {
    yield* input.plugin.trigger(
      "experimental.chat.system.transform",
      { sessionID: input.sessionID, model: input.model },
      { system },
    )
  }
  if (system.length > 2 && system[0] === header) {
    const rest = system.slice(1)
    system.length = 0
    system.push(header, rest.join("\n"))
  }

  const variant =
    !input.small && input.model.variants && input.user.model.variant
      ? input.model.variants[input.user.model.variant]
      : {}
  const base = input.small
    ? ProviderTransform.smallOptions(input.model)
    : ProviderTransform.options({
        model: input.model,
        sessionID: input.sessionID,
        providerOptions: input.provider.options,
      })
  const options = mergeOptions(mergeOptions(mergeOptions(base, input.model.options), input.agent.options), variant)
  if (
    input.model.api.npm === "@ai-sdk/azure" &&
    (input.provider.options.useCompletionUrls || input.model.options.useCompletionUrls || options.useCompletionUrls)
  ) {
    delete options.reasoningSummary
    delete options.include
  }
  if (isOpenaiOauth) options.instructions = system.join("\n")

  const messages =
    isOpenaiOauth || input.isWorkflow
      ? input.messages
      : [
          ...system.map(
            (x): ModelMessage => ({
              role: "system",
              content: x,
            }),
          ),
          ...(simple ? simpleTranscript(input.messages) : input.messages),
        ]

  const params = yield* input.plugin.trigger(
    "chat.params",
    {
      sessionID: input.sessionID,
      agent: input.agent.name,
      model: input.model,
      provider: input.provider,
      message: input.user,
    },
    {
      temperature: input.model.capabilities.temperature
        ? (input.agent.temperature ?? ProviderTransform.temperature(input.model))
        : undefined,
      topP: input.agent.topP ?? ProviderTransform.topP(input.model),
      topK: ProviderTransform.topK(input.model),
      maxOutputTokens: ProviderTransform.maxOutputTokens(input.model, input.flags.outputTokenMax),
      options,
    },
  )

  const { headers } = yield* input.plugin.trigger(
    "chat.headers",
    {
      sessionID: input.sessionID,
      agent: input.agent.name,
      model: input.model,
      provider: input.provider,
      message: input.user,
    },
    {
      headers: {},
    },
  )

  const tools = resolveTools(input, simple)
  // Simple mode targets weak/local models with small context windows. The full builtin tool
  // descriptions run ~3.8k tokens (bash alone is ~1.2k), which can crowd out the response on a
  // small num_ctx. Replace them with terse descriptions; the parameter schemas still convey the
  // argument names/types. Only builtin tools are overridden; custom/MCP tools keep their own text.
  if (simple) {
    for (const key of Object.keys(tools)) {
      const short = SIMPLE_TOOL_DESCRIPTIONS[key]
      if (short) tools[key] = { ...tools[key], description: short }
    }
  }
  // Codex parity: OpenAI Responses-family providers hardcode `strict: false`
  // on every function tool so MCP-sourced and dynamic schemas that don't
  // satisfy OpenAI's structured-outputs constraints still register.
  if (
    input.model.api.npm === "@ai-sdk/openai" ||
    input.model.api.npm === "@ai-sdk/azure" ||
    input.model.api.npm === "@ai-sdk/amazon-bedrock/mantle"
  ) {
    for (const key of Object.keys(tools)) tools[key] = { ...tools[key], strict: false }
  }
  if (
    input.model.providerID.includes("github-copilot") &&
    Object.keys(tools).length === 0 &&
    hasToolCalls(input.messages)
  ) {
    // Copilot needs a tools field when replaying prior tool calls, even if no tools are currently enabled.
    tools["_noop"] = aiTool({
      description: "Do not call this tool. It exists only for API compatibility and must never be invoked.",
      inputSchema: jsonSchema({
        type: "object",
        properties: {
          reason: { type: "string", description: "Unused" },
        },
      }),
      execute: async () => ({ output: "", title: "", metadata: {} }),
    })
  }

  const opencodeProjectID = input.model.providerID.startsWith("opencode")
    ? (yield* InstanceState.context).project.id
    : undefined

  return {
    system,
    messages,
    tools: Object.fromEntries(Object.entries(tools).toSorted(([a], [b]) => a.localeCompare(b))),
    params,
    messageTransformOptions: options,
    headers: {
      ...(input.model.providerID.startsWith("opencode")
        ? {
            ...(opencodeProjectID ? { "x-opencode-project": opencodeProjectID } : {}),
            "x-opencode-session": input.sessionID,
            "x-opencode-request": input.user.id,
            "x-opencode-client": input.flags.client,
            "User-Agent": USER_AGENT,
          }
        : {
            "x-session-affinity": input.sessionID,
            "X-Session-Id": input.sessionID,
            ...(input.parentSessionID ? { "x-parent-session-id": input.parentSessionID } : {}),
            "User-Agent": USER_AGENT,
          }),
      ...input.model.headers,
      ...headers,
    },
  }
})

function simplePrompt(agent: Agent.Info, model: Provider.Model) {
  if (agent.name === "explore") return [PROMPT_SIMPLE_EXPLORE]
  if (agent.name === "summary") return [PROMPT_SIMPLE_SUMMARY]
  if (agent.name === "title") return [PROMPT_SIMPLE_TITLE]
  if (agent.name === "compaction") return [PROMPT_SIMPLE_COMPACTION]
  return SystemPrompt.provider(model)
}

function simpleTranscript(messages: ModelMessage[]): ModelMessage[] {
  const last = messages.at(-1)
  const history = last?.role === "user" ? messages.slice(0, -1) : messages
  const current = last?.role === "user" ? simpleMessageContent(last) : "Continue from the last tool result. Answer the user's request."
  return [
    {
      role: "user",
      content: [
        "Conversation so far:",
        history.length === 0 ? "No previous messages." : history.map(simpleTranscriptLine).join("\n\n"),
        "",
        "Current user request:",
        current,
      ].join("\n"),
    },
  ]
}

function simpleTranscriptLine(message: ModelMessage) {
  const role = message.role === "assistant" ? "Assistant" : message.role === "user" ? "User" : message.role
  return `${role}: ${simpleMessageContent(message)}`
}

function simpleMessageContent(message: ModelMessage) {
  return typeof message.content === "string" ? message.content : JSON.stringify(message.content)
}

function resolveTools(input: Pick<PrepareInput, "tools" | "agent" | "permission" | "user">, simple: boolean) {
  const disabled = Permission.disabled(
    Object.keys(input.tools),
    Permission.merge(input.agent.permission, input.permission ?? []),
  )
  return Record.filter(input.tools, (_, k) => {
    if (input.user.tools?.[k] === false || disabled.has(k)) return false
    // These tools are unreliable for the weak models simplePrompt targets: task orchestrates
    // other agents, skill has no context in simple mode (skill instructions are stripped), and
    // todowrite invites busywork simple mode already discourages. Hide them unless the model
    // config explicitly re-enables the tool via `tools: { <id>: true }`.
    if (simple && SIMPLE_HIDDEN_TOOLS.has(k) && input.user.tools?.[k] !== true) return false
    return true
  })
}

const SIMPLE_HIDDEN_TOOLS = new Set(["task", "skill", "todowrite"])

// Terse builtin tool descriptions for simplePrompt mode. Keyed by tool id. The parameter schema
// still tells the model the argument names/types, so these only need to convey purpose plus the
// one or two rules that matter for a weak model.
const SIMPLE_TOOL_DESCRIPTIONS: Record<string, string> = {
  bash: "Run a shell command in the working directory (git, npm, docker, build/test, etc.). Use the `workdir` parameter instead of `cd`. For reading, writing, searching, or finding files, use the dedicated tools (read/write/edit/glob/grep), not bash.",
  read: "Read a file (absolute `filePath`) or list a directory. Returns up to 2000 lines from `offset`; call again with a larger offset for more, or use grep to search large files. Can also read images/PDFs.",
  edit: "Edit a file: replace `oldString` with `newString` in `filePath`. `oldString` must match the file exactly and be unique (include surrounding context); set `replaceAll` to replace every occurrence.",
  write: "Create or overwrite a file at `filePath` with `content`. Read an existing file before overwriting it.",
  glob: "Find files by name/glob `pattern` (e.g. `**/*.ts`), optionally under `path`. Returns matching file paths. Use this instead of bash find/ls.",
  grep: "Search file contents by regex `pattern`, optionally scoped by `path`/`include`. Returns matching files and lines. Use this instead of bash grep.",
  webfetch: "Fetch a `url` and return its contents as text, markdown, or html.",
  question: "Ask the user a clarifying question (`questions`) with options. Use only when you genuinely need input to proceed.",
}

export function hasToolCalls(messages: ModelMessage[]): boolean {
  for (const msg of messages) {
    if (!Array.isArray(msg.content)) continue
    for (const part of msg.content) {
      if (part.type === "tool-call" || part.type === "tool-result") return true
    }
  }
  return false
}

export * as LLMRequestPrep from "./request"
