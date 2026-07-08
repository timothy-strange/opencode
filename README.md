<p align="center">
  <a href="https://opencode.ai">
    <picture>
      <source srcset="packages/console/app/src/asset/logo-ornate-dark.svg" media="(prefers-color-scheme: dark)">
      <source srcset="packages/console/app/src/asset/logo-ornate-light.svg" media="(prefers-color-scheme: light)">
      <img src="packages/console/app/src/asset/logo-ornate-light.svg" alt="OpenCode logo">
    </picture>
  </a>
</p>
<p align="center">A fork of <a href="https://github.com/anomalyco/opencode">OpenCode</a> exploring a simplified prompt mode for weak/local models.</p>

---

### About this fork

This is a personal fork of [anomalyco/opencode](https://github.com/anomalyco/opencode), the open source AI coding agent. It is **not affiliated with the OpenCode team** and is not kept in sync with upstream.

For installation instructions, the desktop app, built-in agents, general documentation, and everything else about OpenCode itself, see the [upstream README](https://github.com/anomalyco/opencode) and the [official docs](https://opencode.ai/docs).

This fork exists to hold one experiment: a `simplePrompt` mode that lets OpenCode drive weaker or local tool-calling models (e.g. via Ollama or another OpenAI-compatible endpoint). It's no longer under active development here — local models tried against it weren't capable enough to justify the extra complexity — but the code and config are left in place (on the `feat/simple-prompt-mode` branch, now this repo's default branch) in case it's useful as a starting point for anyone else.

### Simple Prompt Mode (local/weak models)

When `simplePrompt` is enabled for a model, OpenCode swaps in a short base prompt, flattens the conversation into a single simplified transcript, and trims tool output/history to fit a small context window, instead of using the default prompts and full history that assume a strong tool-calling model.

Configure it per model in `opencode.json`:

```jsonc
{
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:11434/v1" },
      "models": {
        "qwen2.5-coder:7b": {
          "simplePrompt": true,
          "localModel": true,
          "localModelStrength": "medium",
          "simpleContext": {
            "historyTokens": 1200,
            "budgetTokens": 1000,
            "firstUserMaxChars": 400,
            "toolOutputMaxChars": 300,
            "currentTurnRecentSteps": 2
          }
        }
      }
    }
  },
  "simpleInstructions": ["AGENTS.simple.md"]
}
```

Options, all set per-model under `provider.<name>.models.<id>`:

- **`simplePrompt`** (`boolean`) - Use the simplified prompt/history mode for this model: a short base prompt, flattened conversation history, and slim reminders instead of the full default prompt.
- **`localModel`** (`boolean`) - Marks the model as local. OpenCode will automatically pick the strongest model flagged `localModel` and register it as a read-only `local-explore` subagent, used by default for simple bounded context-gathering tasks (glob/grep/read only - no edits, bash, or web access) so a slow/expensive main model isn't spent on simple lookups. If you define your own `local-explore` agent in config, it's left untouched.
- **`localModelStrength`** (`"small" | "medium" | "strong"`) - Relative capability of the local model, used only to pick which local model becomes `local-explore` when you have more than one configured.
- **`simpleContext`** - Per-model overrides for the context-shaping limits applied in simple prompt mode:
  - **`historyTokens`** - Max token budget for the flattened, older transcript history sent to the model.
  - **`budgetTokens`** - Approximate token budget used when selecting which older messages to keep.
  - **`firstUserMaxChars`** - Max characters retained from the first user message in the conversation.
  - **`toolOutputMaxChars`** - Base max characters retained per tool output.
  - **`currentTurnRecentSteps`** - Number of recent steps in the current turn protected from being trimmed/omitted.

You can also add `simpleInstructions` (top-level, alongside `instructions`) to point at instruction files (e.g. `AGENTS.simple.md`) that should only be loaded for models running in simple prompt mode, kept separate from your normal `AGENTS.md`/`CLAUDE.md`.

---

**Upstream project:** [anomalyco/opencode](https://github.com/anomalyco/opencode) | [Discord](https://opencode.ai/discord) | [opencode.ai](https://opencode.ai)
