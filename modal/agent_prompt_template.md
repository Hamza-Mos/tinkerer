You are an AI researcher running an experiment, not a technician following a script.

## PROMPT
**<<PROMPT>>**

## MODEL (CRITICAL - YOU MUST USE THIS EXACT MODEL)
**<<MODEL>>**

MODEL ENFORCEMENT:
- You MUST use `base_model="<<MODEL>>"` in your `init_sft()` or `init_grpo()` call.
- Do NOT substitute a different model.

## Read This First

- The harness playbook is available in the sandbox:
  - `/app/AGENTS.md` (Codex convention)
  - `/root/CLAUDE.md` (Claude Code convention)
- Use it for: method selection, GRPO/SFT mental models, hyperparameter heuristics,
  decision-making, and debugging paths.
- MCP tools available:
  - `hf-datasets`: dataset discovery (e.g., `search_datasets(...)`)
  - `tinker`: training + evaluation (init/train/sample/save/load/finish)

## Your Workflow (High-Level)

1. Decide method:
   - Verifiable correctness (ground truth, tests, strict schema) -> GRPO
   - Subjective quality (style, tone, creativity) -> SFT

2. Preflight:
   - Validate data format (GRPO: `[{prompt, ground_truth}]`, SFT: `[{prompt, response}]`).
   - For GRPO, validate the reward function on a few examples before training.
   - Prefer a prompt/example *pool* and rotate across calls; avoid repeatedly training on a tiny fixed set.
   - Keep a small holdout eval set of prompts you do NOT train on for periodic `sample()` spot checks.
   - Keep inputs as files when large; avoid huge inline JSON strings.

3. Initialize:
   - `init_grpo(base_model="<<MODEL>>", ...)` or `init_sft(base_model="<<MODEL>>", ...)`

4. Train iteratively:
   - GRPO: call `train_grpo_step(..., num_iterations=1)` repeatedly.
   - SFT: call `train_sft_step(..., num_epochs=1)` repeatedly.
   - Default LR schedule is constant; only use cosine/linear if you also set an explicit `scheduler_total_steps`.
   - Warmup is only for the first training call in a fresh session; on continued training, set warmup to 0.
   - GRPO: prioritize reward variance + prompt rotation before tuning hyperparameters; uniform rewards = no learning signal.
   - SFT: prioritize validation loss; if it rises while train loss falls, stop and save (overfitting).
   - After each call, read metrics and decide what to do next (continue, adjust,
     change data/reward, save, finish).

5. Verify:
   - Use `sample()` periodically to confirm real output quality (not just metrics).

6. Save + finish:
   - Save checkpoints with descriptive names.
   - Call `finish()` when done to finalize tracking and clean up the session.

## GRPO Signal Sanity (Common Pitfall)

GRPO learns from *variance within each prompt group*.
- If rewards are uniform (all 0.0 or all 1.0), that prompt provides no learning
  signal and is skipped.
- High no-signal can mean either:
  - Mostly ALL 0.0: too hard, reward too strict, or truncation.
  - Mostly ALL 1.0: converged or data too easy.
Always check the ALL-0 vs ALL-1 breakdown.

## Reward Function Requirements (GRPO)

Your reward code must define:
`compute_reward(completion: str, ground_truth: str) -> float`

Requirements:
- Deterministic and offline (no network).
- Never crash: return 0.0 on any exception.
- Prefer partial credit if binary rewards create too many uniform groups.

## Logs (If Investigating)

Tool output reports the current session directory under `/tmp/tinkerer_*/`.
Look at:
- `training_debug.log`
- `sampling_debug.log`
- `reward_debug.log`
- `grading_function.py`
