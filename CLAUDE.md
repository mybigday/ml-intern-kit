# CLAUDE.md — ml-intern-kit

The operating rules for this project live in `AGENTS.md` (provider-neutral) so
the same instructions drive Claude Code, Codex, Cursor, and any other
AGENTS.md-aware tool. Read that first.

@AGENTS.md

## Claude-Code-specific notes

- **Skills** in `.claude/skills/` codify the recipes. The model-invoked
  descriptions are tuned so the right skill fires automatically:
  - `research-recipe` — literature-first crawl before any ML code
  - `inspect-dataset` — column/format audit on a HF dataset
  - `train-sft` — supervised fine-tune with TRL `SFTTrainer` (CUDA/ROCm/MPS/CPU)
  - `train-mlx` — native LoRA fine-tune on Apple Silicon via mlx-lm
  - `rocm-strix-halo` — gfx1151 / Ryzen AI Max+ 395 setup (TheRock nightly)
  - `train-dpo` — DPO preference optimization
  - `launch-hf-job` — pre-flight + `hf jobs run` submission
  - `eval-model` — `lm-eval-harness` wrapper
  - `env-bootstrap` — recreate the env on a new machine

- **Sub-agents.** For literature crawls, spawn `Agent(subagent_type="Explore",
  prompt=<see AGENTS.md research-sub-agent prompt>)` so raw paper/doc bytes
  never enter the main context.

- **Permissions.** `.claude/settings.json` already allow-lists `uv`,
  `python`, `pip`, `hf`, `huggingface-cli`, `wandb`, `accelerate`, `gh search`,
  `git`, and the local `scripts/*.py`. Anything destructive (`rm -rf`, force
  push, dropping a Hub repo) still prompts.

- **House style.** Terse. No "Great question!". Hub URLs whenever you mention
  a model/dataset/job. Logging discipline (`disable_tqdm=True`, plain-text loss
  lines) is non-negotiable for training scripts.
