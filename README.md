# Tinkerer

Agent-driven SFT + GRPO on the Tinker API, designed to run inside a Modal sandbox.
Claude Code is the default agent, with optional OpenAI Codex support.

## Quick Start

### Prereqs

- Python 3.13 (repo pins assume 3.13)
- Modal CLI authenticated (`modal login`)

### Install

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode (required for scripts and tests)
pip install -e .
```

### Modal Secrets

Create these once in your Modal account:

```bash
# Required for all runs.
# You can optionally include WANDB_API_KEY and HF_TOKEN here for observability and gated datasets.
modal secret create tinker-api-key TINKER_API_KEY=... WANDB_API_KEY=... HF_TOKEN=...

# Required for the default agent (Claude Code).
modal secret create anthropic-api-key ANTHROPIC_API_KEY=...

# Required for Codex runs.
modal secret create openai-api-key OPENAI_API_KEY=...
```

### Local `.env` (Optional)

For local scripts (like `python scripts/checkpoint_compare.py` or `python scripts/live_model_matrix_smoke.py`), you can keep
environment variables in a local `.env` file.

- Example template: `.env.example`
- Your real `.env` is ignored by git.

Modal runs do not automatically read `.env`; secrets must be added with `modal secret create ...`.

Codex note:

- This harness uses API-key auth via `OPENAI_API_KEY`.
- `gpt-5.3-codex` is not available via the OpenAI API for most accounts (you'll get `model_not_found`), so we default to `gpt-5.2-codex`.

### Run A Demo

Notes:

- Omit `MODAL_FORCE_BUILD=1` unless you changed repo code and need Modal to rebuild the image.
- `TINKERER_DEBUG` defaults to `1` in the sandbox; set it to `0` for less logging.
- `--model` and `--prompt` are required for non-`--test-only` runs.

```bash
# Claude Code (default)
modal run modal/run_tinkerer.py \
  --model "meta-llama/Llama-3.2-3B" \
  --prompt "$(cat prompts/arithmetic_quick_win.txt)"

# Codex
modal run modal/run_tinkerer.py \
  --agent codex \
  --model "meta-llama/Llama-3.2-3B" \
  --prompt "$(cat prompts/arithmetic_quick_win.txt)"
```

More demos: see `prompts/`.

## Quickstart Demos

These are copy/paste-friendly "known good" commands.
They include `MODAL_FORCE_BUILD=1` for the "I'm actively hacking" workflow; omit it for faster startup when you haven't
changed repo code.

```bash
# 0) Quick infra check
modal run modal/run_tinkerer.py --test-only

# 1) Fast "wow" RL demo (30-90 min)
MODAL_FORCE_BUILD=1 TINKERER_DEBUG=1 modal run modal/run_tinkerer.py \
  --model "meta-llama/Llama-3.2-3B" \
  --prompt "$(cat prompts/arithmetic_quick_win.txt)"

# 2) MBPP few-hour RL demo
MODAL_FORCE_BUILD=1 TINKERER_DEBUG=1 modal run modal/run_tinkerer.py \
  --model "Qwen/Qwen3-8B-Base" \
  --prompt "$(cat prompts/mbpp_fast_8b.txt)"

# 3) GSM8K few-hour RL demo
MODAL_FORCE_BUILD=1 TINKERER_DEBUG=1 modal run modal/run_tinkerer.py \
  --model "Qwen/Qwen3-8B-Base" \
  --prompt "$(cat prompts/gsm8k_fast_8b.txt)"

# 4) SFT few-hour demo
MODAL_FORCE_BUILD=1 TINKERER_DEBUG=1 modal run modal/run_tinkerer.py \
  --model "Qwen/Qwen3-8B-Base" \
  --prompt "$(cat prompts/sft_norobots_8b.txt)"

# 5) MBPP GRPO demo (recommended model in prompt: Qwen3-30B-A3B-Base)
MODAL_FORCE_BUILD=1 TINKERER_DEBUG=1 modal run modal/run_tinkerer.py \
  --model "Qwen/Qwen3-30B-A3B-Base" \
  --prompt "$(cat prompts/mbpp.txt)"

# 6) GSM8K GRPO demo (recommended model in prompt: Qwen3-30B-A3B-Base)
MODAL_FORCE_BUILD=1 TINKERER_DEBUG=1 modal run modal/run_tinkerer.py \
  --model "Qwen/Qwen3-30B-A3B-Base" \
  --prompt "$(cat prompts/math_gsm8k.txt)"

# 7) AIME GRPO demo (prompt recommends Qwen3-30B-A3B-Base OR DeepSeek-V3.1-Base)
MODAL_FORCE_BUILD=1 TINKERER_DEBUG=1 modal run modal/run_tinkerer.py \
  --model "Qwen/Qwen3-30B-A3B-Base" \
  --prompt "$(cat prompts/math_aime.txt)"

# 7b) AIME with DeepSeek base variant
MODAL_FORCE_BUILD=1 TINKERER_DEBUG=1 modal run modal/run_tinkerer.py \
  --model "deepseek-ai/DeepSeek-V3.1-Base" \
  --prompt "$(cat prompts/math_aime.txt)"
```

## Prompts

Prompts live under `prompts/*.txt`. These are _recipes_, not rigid scripts.

Design intent:

- `modal/sandbox_agent_instructions.md` is the general agent playbook.
- `prompts/*.txt` are lightweight, task-oriented recipes that guide decision
  making without hard-coding a specific dataset, pool size, or huge reward
  function blob.

| File                       | Method | Intended use                                        |
| -------------------------- | ------ | --------------------------------------------------- |
| `general_template.txt`     | GRPO   | Template for any verifiable task                    |
| `mbpp.txt`                 | GRPO   | Verifiable code generation (unit-test style)        |
| `mbpp_fast_8b.txt`         | GRPO   | Same as above, biased toward smaller/cheaper models |
| `math_gsm8k.txt`           | GRPO   | Verifiable math reasoning (short final answers)     |
| `gsm8k_fast_8b.txt`        | GRPO   | Same as above, biased toward smaller/cheaper models |
| `math_aime.txt`            | GRPO   | Hard math reasoning (low baseline pass rates)       |
| `arithmetic_quick_win.txt` | GRPO   | Fast synthetic "wow" demo (verifiable arithmetic)   |
| `sft_norobots_8b.txt`      | SFT    | Instruction/style SFT recipe (subjective quality)   |

Notes:

- The `--model` CLI flag is the source of truth; prompts should not try to
  override it.
- W&B and HF dataset access are enabled automatically when the corresponding
  env vars are present in the sandbox (for example by including `WANDB_API_KEY`
  and/or `HF_TOKEN` in your `tinker-api-key` Modal secret).

## What You're Running

`modal/run_tinkerer.py` builds an image that installs:

- `@anthropic-ai/claude-code` (Claude Code CLI)
- `@openai/codex` (Codex CLI)

The agent talks to the MCP server (`python -m tinker_mcp`) via `.mcp.json`.

### Architecture

At a high level, you are running an agent loop inside a Modal container. The
agent calls MCP tools (dataset + training) and the Tinker API performs remote
training.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Modal Container (sandbox)                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 Agent CLI (Claude / Codex)                 │  │
│  │  - Reads MCP config                                        │  │
│  │  - Calls tools (sample/train/save/finish)                  │  │
│  └───────────────────┬───────────────────────────────────────┘  │
│                      │ MCP Protocol                              │
│  ┌───────────────────┴───────────────────────────────────────┐  │
│  │                  MCP Servers                               │  │
│  │  ┌─────────────────┐  ┌─────────────────────────────────┐ │  │
│  │  │  hf-datasets    │  │    tinker (python -m tinker_mcp) │ │  │
│  │  │  (dataset I/O)  │  │    init_sft/init_grpo            │ │  │
│  │  │                 │  │    train_sft_step/train_grpo_step│ │  │
│  │  │                 │  │    sample/save/load/finish       │ │  │
│  │  └─────────────────┘  └─────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                    Tinker API (remote training)                  │
└──────────────────────────────┼───────────────────────────────────┘
                               ▼
                      Cloud Training Backend
```

## MCP Server

The MCP server lives in `mcp-servers/tinker_mcp/` and is intentionally stateful
so it can manage a long-lived training session across multiple tool calls.

Entry points:

- Tools + docstrings: `mcp-servers/tinker_mcp/server.py`
- Model registry: `mcp-servers/tinker_mcp/models.py`
- Training logic: `mcp-servers/tinker_mcp/training/`

Extending:

- Add a model: add/update `MODEL_INFO` in `mcp-servers/tinker_mcp/models.py`
  and ensure the renderer mapping supports it.
- Add a method: implement in `mcp-servers/tinker_mcp/training/` and expose a new
  tool in `mcp-servers/tinker_mcp/server.py`.

## Best Practices (GRPO + SFT)

### Choose The Right Method

- GRPO: you can verify correctness from `(completion, ground_truth)` (tests, exact answers, parsable structured output).
- SFT: "good" is subjective (style/tone) and you have high-quality examples.

### Don't "Pick A Magic Step Count"

Treat training as an iterative control loop:

- Run small steps.
- Look at metrics _and_ `sample()` outputs.
- Stop when additional steps aren't changing behavior.

The Tinker docs include examples that improve over ~15 RL iterations in a minimal GSM8K script, but that number is task/data/reward dependent. Use it only as a sanity baseline, not a requirement.

### GRPO Signal: Variance Matters

GRPO learns from _within-prompt_ variance across a group of samples:

- ALL 0.0 rewards: too hard / reward too strict / truncation.
- ALL 1.0 rewards: converged or data too easy.

Both cases produce "no-signal" and are expected; the job is to keep enough "mixed" prompts in the batch.

### Model Choice

For Tinker-specific guidance on which model families to use (Base vs Instruction vs Hybrid/Reasoning), see the official model lineup docs and follow their recommendations.

## W&B: How To Read The Charts

- One Modal run == one W&B _run_.
- GRPO iterations are points on that run's charts (x-axis is `iteration`).
- If a chart shows multiple colored lines, you have multiple runs selected in the UI.
- Use W&B "groups" to organize comparable runs (same prompt/model, different seeds/agents).

This harness defaults `WANDB_GROUP` so parallel Claude/Codex runs land in the same group while keeping run names unique.

If you're comparing Claude vs Codex: keep everything else fixed (model, prompts/pool, reward function, LR schedule,
temperature, group size, LoRA rank). Otherwise you're measuring the setup, not the agent.

## Debugging

Inside the sandbox, logs live under `/tmp/tinkerer_*/`:

- `init_debug.log`, `training_debug.log`, `sampling_debug.log`, `reward_debug.log`, `wandb_debug.log`

Training also writes a watchdog heartbeat: `/tmp/tinkerer_progress_*.txt`.

### Debug Workflow (Copy/Paste)

This is intentionally copy/paste friendly.

<details>
<summary>Show the full debugging workflow</summary>

#### Step 1: Get Container ID + Session Directory

```bash
modal container list

# Pick the container you care about.
CID="ta-..."

# Most logs live under a session-specific directory.
SDIR=$(modal container exec "$CID" -- bash -lc 'ls -td /tmp/tinkerer_*/ 2>/dev/null | head -1')
echo "Session dir: $SDIR"
```

Notes:

- `modal container exec` requires `--` before the command.
- The global progress heartbeat is in `/tmp/tinkerer_progress_*.txt` (PID in name).
- The session-scoped progress heartbeat is `progress.txt` inside the session dir.

#### Step 2: Stream Live Agent Output

```bash
# Streams logs live (Ctrl-C to stop).
modal container logs --timestamps "$CID"

# Snapshot a few seconds of logs without a long-running stream.
# If `timeout` is unavailable, omit it and Ctrl-C manually.
timeout 5s modal container logs --timestamps "$CID" 2>&1 | tail -200
```

#### Step 3: Check Initialization

```bash
modal container exec "$CID" -- bash -lc 'tail -n 200 '"$SDIR"'/init_debug.log 2>/dev/null || true'
modal container exec "$CID" -- bash -lc 'tail -n 200 '"$SDIR"'/wandb_debug.log 2>/dev/null || true'
```

#### Step 4: Monitor Training Progress

Progress heartbeat:

```bash
modal container exec "$CID" -- bash -lc 'ls -la /tmp/tinkerer_progress_*.txt 2>/dev/null || true'
modal container exec "$CID" -- bash -lc 'tail -n 200 /tmp/tinkerer_progress_*.txt 2>/dev/null || true'
```

Session progress:

```bash
modal container exec "$CID" -- bash -lc 'tail -n 200 '"$SDIR"'/progress.txt 2>/dev/null || true'
```

Training loop details:

```bash
modal container exec "$CID" -- bash -lc 'tail -n 200 '"$SDIR"'/training_debug.log 2>/dev/null || true'
modal container exec "$CID" -- bash -lc 'grep -E \"\\[ITER|DONE:|WARNING|ERROR\" '"$SDIR"'/training_debug.log 2>/dev/null | tail -n 50 || true'
```

#### Step 5: Inspect Reward Function (GRPO only)

```bash
modal container exec "$CID" -- bash -lc 'cat '"$SDIR"'/grading_function.py 2>/dev/null || true'
modal container exec "$CID" -- bash -lc 'cat '"$SDIR"'/grading_validation.log 2>/dev/null || true'
modal container exec "$CID" -- bash -lc 'tail -n 200 '"$SDIR"'/reward_debug.log 2>/dev/null || true'
```

#### Step 6: Inspect Generations

```bash
modal container exec "$CID" -- bash -lc 'tail -n 200 '"$SDIR"'/sampling_debug.log 2>/dev/null || true'
```

#### Step 7: Error Detection

```bash
modal container exec "$CID" -- bash -lc 'grep -iE \"error|failed|timeout|exception\" '"$SDIR"'/training_debug.log 2>/dev/null | tail -n 50 || true'
modal container exec "$CID" -- bash -lc 'grep -iE \"error|failed|timeout|exception\" '"$SDIR"'/reward_debug.log 2>/dev/null | tail -n 50 || true'
```

#### Step 8: Check Checkpoints

```bash
modal container exec "$CID" -- bash -lc 'tail -n 200 '"$SDIR"'/save_load_debug.log 2>/dev/null || true'
```

#### Quick Status Check (All At Once)

```bash
modal container exec "$CID" -- bash -lc '
  SDIR=$(ls -td /tmp/tinkerer_*/ 2>/dev/null | head -1)
  echo \"=== SESSION DIR ===\" && echo \"$SDIR\"
  echo \"=== INIT (tail) ===\" && tail -n 10 \"$SDIR/init_debug.log\" 2>/dev/null || true
  echo \"=== W&B (tail) ===\" && tail -n 10 \"$SDIR/wandb_debug.log\" 2>/dev/null || true
  echo \"=== PROGRESS (tail) ===\" && tail -n 12 /tmp/tinkerer_progress_*.txt 2>/dev/null || true
  echo \"=== TRAINING (tail) ===\" && tail -n 25 \"$SDIR/training_debug.log\" 2>/dev/null || true
  echo \"=== WARN/ERR (recent) ===\" && grep -iE \"warning|error|timeout\" \"$SDIR/training_debug.log\" 2>/dev/null | tail -n 15 || true
'
```

</details>

## Project Structure

```
tinkerer/
  mcp-servers/
    tinker_mcp/              # MCP server + training logic
  modal/                     # Modal runner (agent orchestration)
  prompts/                   # Prompts (demo recipes)
  scripts/                   # Local utilities (smoke tests, checkpoint compare)
  tests/                     # Unit tests
  requirements.txt           # Python dependencies (pinned)
  pyproject.toml             # Package metadata
```

## Environment Variables

| Variable                                        | Required | Description                                                                                                                     |
| ----------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `TINKER_API_KEY`                                | Yes      | API key for the Tinker training service.                                                                                        |
| `WANDB_API_KEY`                                 | No       | Weights & Biases API key for training monitoring.                                                                               |
| `HF_TOKEN`                                      | No       | HuggingFace token for accessing gated datasets.                                                                                 |
| `TINKERER_DEBUG`                                | No       | Set to "1" for verbose debug logging to session directory                                                                       |
| `TINKERER_GRPO_SAMPLING_DEBUG_PROMPT_LIMIT`     | No       | Number of prompts with detailed GRPO sampling debug (default: `3`, `-1` = all prompts, `0` = disable per-prompt sampling debug) |
| `TINKERER_GRPO_AUTOCHECKPOINT_REWARD_THRESHOLD` | No       | Auto-checkpoint reward threshold for GRPO (default: `0.3`)                                                                      |
| `TINKERER_GRPO_AUTOCHECKPOINT_MIN_ITERATIONS`   | No       | Minimum cumulative GRPO iterations before auto-checkpoint can trigger (default: `3`)                                            |
| `TINKERER_TOOL_LOCK_TIMEOUT_SECONDS`            | No       | Max time a concurrent MCP tool call waits for the singleton-session lock before returning busy (default: `10.0`, `-1` waits indefinitely) |
| `ANTHROPIC_API_KEY`                             | No       | Required when running with Claude Code via Modal deployment                                                                     |
| `OPENAI_API_KEY`                                | No       | Required when running with Codex via Modal deployment                                                                           |
| `CODEX_MODEL`                                   | No       | Codex model override (default: `gpt-5.2-codex`)                                                                                 |
| `CODEX_REASONING_EFFORT`                        | No       | Codex reasoning effort (default: `xhigh`)                                                                                       |
| `CODEX_REASONING_SUMMARY`                       | No       | Codex reasoning summary verbosity (default: `detailed`)                                                                         |

## Testing a Checkpoint

```bash
# Compare base model vs fine-tuned checkpoint
export TINKER_CHECKPOINT="tinker://your-checkpoint-path"
export TINKER_BASE_MODEL="meta-llama/Llama-3.2-1B"
python scripts/checkpoint_compare.py
```

## Live Matrix Smoke (SFT + GRPO)

Run a low-cost live compatibility matrix across model families:

```bash
python scripts/live_model_matrix_smoke.py \
  --families "base,instruction,hybrid,reasoning" \
  --methods "sft,grpo" \
  --max-tokens 96 \
  --lora-rank 8 \
  --group-size 4
```

GitHub Actions workflow: `.github/workflows/live-model-matrix-smoke.yml`

- Uses a per-family matrix (`base`, `instruction`, `hybrid`, `reasoning`)
- Runs on manual dispatch and weekly schedule
- Requires `TINKER_API_KEY` repo secret
- Note: Instruction-tuned models may produce uniform rewards in smoke tests, causing GRPO to skip samples (expected behavior). Use `--allow-failures=1` if needed.
- Uploads per-family JSON smoke reports as artifacts

## Development

```bash
# Run tests (pytest is configured to find packages in mcp-servers/)
python -m pytest -q

# Lint
flake8
```

**Note:** The package must be installed in editable mode (`pip install -e .`) for tests to run. If you encounter `ModuleNotFoundError: No module named 'tinker_mcp'`:
- Ensure you've run `pip install -e .` 
- Use `python -m pytest` instead of `pytest` directly
- The `pyproject.toml` includes `pythonpath = ["mcp-servers"]` to help pytest find the package
