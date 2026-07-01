import { describe, expect, test } from "bun:test"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ContextShaping } from "../../src/session/context-shaping"
import { MessageID, PartID, SessionID } from "../../src/session/schema"

const sessionID = SessionID.make("session")
const providerID = ProviderV2.ID.make("test")

function user(id: string, text: string): SessionV1.WithParts {
  const messageID = MessageID.make(id)
  return {
    info: {
      id: messageID,
      sessionID,
      role: "user",
      time: { created: 0 },
      agent: "build",
      model: { providerID, modelID: ModelV2.ID.make("test") },
      tools: {},
      mode: "",
    } as unknown as SessionV1.User,
    parts: [
      {
        id: PartID.make(`prt_${id}`),
        sessionID,
        messageID,
        type: "text",
        text,
      } as SessionV1.TextPart,
    ],
  }
}

function assistant(id: string, parentID: string, text: string): SessionV1.WithParts {
  const messageID = MessageID.make(id)
  return {
    info: {
      id: messageID,
      sessionID,
      role: "assistant",
      time: { created: 0 },
      parentID: MessageID.make(parentID),
      modelID: ModelV2.ID.make("test"),
      providerID,
      mode: "build",
      agent: "build",
      path: { cwd: "/", root: "/" },
      cost: 0,
      tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    } as unknown as SessionV1.Assistant,
    parts: [
      {
        id: PartID.make(`prt_${id}`),
        sessionID,
        messageID,
        type: "text",
        text,
      } as SessionV1.TextPart,
    ],
  }
}

function tool(id: string, parentID: string, output: string, name = "bash", filePath = "src/file.ts"): SessionV1.WithParts {
  const messageID = MessageID.make(id)
  return {
    info: {
      id: messageID,
      sessionID,
      role: "assistant",
      time: { created: 0 },
      parentID: MessageID.make(parentID),
      modelID: ModelV2.ID.make("test"),
      providerID,
      mode: "build",
      agent: "build",
      path: { cwd: "/", root: "/" },
      cost: 0,
      tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    } as unknown as SessionV1.Assistant,
    parts: [
      {
        id: PartID.make(`prt_${id}`),
        sessionID,
        messageID,
        type: "tool",
        callID: `call_${id}`,
        tool: name,
        state: {
          status: "completed",
          input: name === "read" ? { filePath } : { command: "yes" },
          output,
          title: name,
          metadata: {},
          time: { start: 0, end: 1 },
        },
      } as SessionV1.ToolPart,
    ],
  }
}

describe("ContextShaping.forSimpleModel", () => {
  test("preserves first user message, recent tail, and current user while omitting the middle", () => {
    const messages = [
      user("msg_01", "initial task"),
      assistant("msg_02", "msg_01", "older answer"),
      user("msg_03", "middle request " + "x".repeat(3_000)),
      assistant("msg_04", "msg_03", "middle answer " + "y".repeat(3_000)),
      user("msg_05", "recent request"),
      assistant("msg_06", "msg_05", "recent answer"),
      user("msg_07", "current request"),
    ]

    const result = ContextShaping.forSimpleModel({
      messages,
      currentUserID: MessageID.make("msg_07"),
      budgetTokens: 50,
      firstUserMaxChars: 100,
      toolOutputMaxChars: 100,
    })

    expect(result.messages.map((message) => message.info.id)).toContain(MessageID.make("msg_01"))
    expect(result.messages.map((message) => message.info.id)).toContain(MessageID.make("msg_07"))
    expect(JSON.stringify(result.messages)).toContain("recent answer")
    expect(JSON.stringify(result.messages)).not.toContain("middle request")
    expect(JSON.stringify(result.messages)).toContain("Earlier conversation omitted")
    expect(result.stats.omittedMessages).toBeGreaterThan(0)
  })

  test("caps the first user message without mutating the original history", () => {
    const first = user("msg_01", "a".repeat(40))
    const current = user("msg_02", "current")

    const result = ContextShaping.forSimpleModel({
      messages: [first, current],
      currentUserID: MessageID.make("msg_02"),
      budgetTokens: 1_000,
      firstUserMaxChars: 10,
      toolOutputMaxChars: 100,
    })

    expect(JSON.stringify(result.messages)).toContain("aaaaaaaaaa")
    expect(JSON.stringify(result.messages)).toContain("First user message truncated")
    expect(JSON.stringify(first)).not.toContain("First user message truncated")
    expect(result.stats.truncatedUserParts).toBe(1)
  })

  test("omits low-value old tool output without mutating the original history", () => {
    const output = "0123456789".repeat(10)
    const noisy = tool("msg_02", "msg_01", output)
    const current = user("msg_03", "current")

    const result = ContextShaping.forSimpleModel({
      messages: [user("msg_01", "initial"), noisy, current],
      currentUserID: MessageID.make("msg_03"),
      budgetTokens: 1_000,
      firstUserMaxChars: 100,
      toolOutputMaxChars: 12,
    })

    expect(JSON.stringify(result.messages)).toContain("Tool output omitted for local model context limit")
    expect(JSON.stringify(noisy)).not.toContain("Tool output truncated for local model context limit")
    expect(result.stats.truncatedToolOutputs).toBe(1)
  })

  test("keeps a larger preview for high-value current-turn tool output", () => {
    const output = "0123456789".repeat(10)
    const current = user("msg_02", "current")
    const recentRead = tool("msg_03", "msg_02", output, "read")

    const result = ContextShaping.forSimpleModel({
      messages: [user("msg_01", "initial"), current, recentRead],
      currentUserID: MessageID.make("msg_02"),
      budgetTokens: 1_000,
      firstUserMaxChars: 100,
      toolOutputMaxChars: 12,
    })

    expect(JSON.stringify(result.messages)).toContain("012345678901234567890123456789012345678901234567")
    expect(JSON.stringify(result.messages)).toContain("Tool output truncated for local model context limit")
    expect(JSON.stringify(result.messages)).not.toContain("Tool output omitted")
    expect(result.stats.truncatedToolOutputs).toBe(1)
  })

  test("scores current failures above old noisy successes", () => {
    const oldBash = ContextShaping.scoreToolPart({
      tool: "bash",
      status: "completed",
      age: 8,
      outputChars: 20_000,
      currentTurn: false,
      compacted: false,
      hasFailureSignal: false,
      hasRecentDuplicate: false,
      inWorkingSet: false,
    })
    const currentFailure = ContextShaping.scoreToolPart({
      tool: "bash",
      status: "completed",
      age: 0,
      outputChars: 20_000,
      currentTurn: true,
      compacted: false,
      hasFailureSignal: true,
      hasRecentDuplicate: false,
      inWorkingSet: false,
    })

    expect(currentFailure).toBeGreaterThan(oldBash)
    expect(oldBash).toBeLessThan(250)
  })

  test("drops superseded read output for the same file", () => {
    const oldRead = tool("msg_02", "msg_01", "old read output " + "o".repeat(10_000), "read", "src/a.ts")
    const newRead = tool("msg_03", "msg_01", "new read output " + "n".repeat(10_000), "read", "src/a.ts")
    const current = user("msg_04", "current")

    const result = ContextShaping.forSimpleModel({
      messages: [user("msg_01", "initial"), oldRead, newRead, current],
      currentUserID: MessageID.make("msg_04"),
      budgetTokens: 10_000,
      firstUserMaxChars: 100,
      toolOutputMaxChars: 12,
    })
    const output = JSON.stringify(result.messages)

    expect(output).toContain("Earlier read of src/a.ts superseded by a newer read")
    expect(output).not.toContain("old read output")
    expect(output).toContain("new read out")
    expect(result.stats.truncatedToolOutputs).toBe(2)
  })

  test("boosts tool output touching a file named in the current request", () => {
    const output = "0123456789".repeat(10)
    const touched = tool("msg_02", "msg_01", output, "read", "src/target.ts")
    const current = user("msg_03", "please finish editing src/target.ts now")

    const result = ContextShaping.forSimpleModel({
      messages: [user("msg_01", "initial"), touched, current],
      currentUserID: MessageID.make("msg_03"),
      budgetTokens: 10_000,
      firstUserMaxChars: 100,
      toolOutputMaxChars: 12,
    })

    // With the working-set boost the read scores high enough to keep a larger preview
    // instead of being omitted.
    expect(JSON.stringify(result.messages)).toContain("Tool output truncated for local model context limit")
    expect(JSON.stringify(result.messages)).not.toContain("Tool output omitted")
  })

  test("scores working-set tool output above the same output off the working set", () => {
    const base = {
      tool: "read",
      status: "completed" as const,
      age: 4,
      outputChars: 5_000,
      currentTurn: false,
      compacted: false,
      hasFailureSignal: false,
      hasRecentDuplicate: false,
    }
    expect(ContextShaping.scoreToolPart({ ...base, inWorkingSet: true })).toBeGreaterThan(
      ContextShaping.scoreToolPart({ ...base, inWorkingSet: false }),
    )
  })

  test("caps long old assistant prose without mutating the original history", () => {
    const chatty = assistant("msg_02", "msg_01", "old reasoning " + "z".repeat(3_000))
    const current = user("msg_03", "current")

    const result = ContextShaping.forSimpleModel({
      messages: [user("msg_01", "initial"), chatty, current],
      currentUserID: MessageID.make("msg_03"),
      budgetTokens: 10_000,
      firstUserMaxChars: 100,
      toolOutputMaxChars: 40,
    })

    expect(JSON.stringify(result.messages)).toContain("Assistant message truncated for local model context limit")
    expect(JSON.stringify(chatty)).not.toContain("Assistant message truncated")
  })

  test("keeps the tail of failure-like tool output", () => {
    const failedLog = tool("msg_02", "msg_01", `${"noise\n".repeat(100)}final error: failed at end`)
    const current = user("msg_03", "current")

    const result = ContextShaping.forSimpleModel({
      messages: [user("msg_01", "initial"), failedLog, current],
      currentUserID: MessageID.make("msg_03"),
      budgetTokens: 1_000,
      firstUserMaxChars: 100,
      toolOutputMaxChars: 12,
    })
    const output = JSON.stringify(result.messages)

    expect(output).toContain("failed at end")
    expect(output).toContain("Tool output truncated for local model context limit")
    expect(output).not.toContain("noise\\nnoise\\nnoise\\nnoise")
  })
})
