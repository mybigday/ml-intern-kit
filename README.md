# ml-intern-kit

A portable, **literature-first** training environment for small language models
on the Hugging Face ecosystem. Adapted from
[`smolagents/ml-intern`](https://huggingface.co/spaces/smolagents/ml-intern) —
distilled into a local Python project + a set of Claude Code skills you can
drop on any machine.

The agent rules (operating manual, pre-flight discipline, error-recovery
playbook) live in [`AGENTS.md`](./AGENTS.md). Read that first.

## What you get

- `AGENTS.md` / `CLAUDE.md` — full operating manual for the agent.
- `.claude/skills/` — model-invoked recipes (`research-recipe`, `train-sft`,
  `inspect-dataset`, `launch-hf-job`, `eval-model`, `env-bootstrap`, …).
- `scripts/` — working TRL templates: SFT, DPO, GRPO, dataset audit, HF Jobs
  launcher, `lm-eval-harness` wrapper.
- `pyproject.toml` + `requirements.txt` + `bootstrap.sh` + `Dockerfile` — three
  ways to recreate the env (uv, pip, Docker).

## Bootstrap (any machine, < 2 minutes)

```bash
git clone https://github.com/mybigday/ml-intern-kit.git && cd ml-intern-kit
bash bootstrap.sh           # detects uv > pip, creates .venv, installs stack
cp .env.example .env        # then fill in HF_TOKEN, optionally WANDB_API_KEY
source .venv/bin/activate
```

What `bootstrap.sh` does:

1. Picks `uv` if present, else falls back to `python -m venv` + `pip`.
2. Pins Python `3.11` (see `.python-version`).
3. Installs the core stack from `requirements.txt`. `flash-attn` and `unsloth`
   are **not** installed by default (they have CUDA/ABI requirements that vary
   per box) — install them with `make flash` / `make unsloth` once.
4. Logs you into the HF Hub if `HF_TOKEN` is set.

## Verify

```bash
make doctor      # prints torch+cuda versions, GPU info, hf whoami
```

## Train your first model (smoke test, CPU-friendly)

```bash
python scripts/inspect_dataset.py trl-lib/Capybara
python scripts/train_sft.py \
    --config configs/sft_default.yaml \
    --max-steps 20 --max-samples 256
```

When that works, scale up locally with `accelerate launch` or push to HF Jobs:

```bash
python scripts/launch_hf_job.py \
    --script scripts/train_sft.py \
    --config configs/sft_default.yaml \
    --hardware a10g-largex2 \
    --timeout 4h
```

## Layout

```
ml-intern-kit/
├── AGENTS.md                  # operating manual (provider-neutral)
├── CLAUDE.md                  # Claude Code stub → @AGENTS.md
├── README.md                  # this file
├── pyproject.toml             # uv-managed; setuptools build backend
├── requirements.txt           # pip-installable equivalent of [project] deps
├── requirements-dev.txt       # pytest, ruff, jupyter, etc.
├── .python-version            # 3.11
├── .env.example
├── bootstrap.sh               # uv-or-pip env setup
├── Dockerfile                 # CUDA 12.4 + uv sync, identical env in a box
├── Makefile                   # env / doctor / train / eval / flash / unsloth
├── configs/
│   └── sft_default.yaml
├── scripts/
│   ├── train_sft.py           # TRL SFTTrainer template
│   ├── train_dpo.py           # TRL DPOTrainer template
│   ├── train_grpo.py          # TRL GRPOTrainer template
│   ├── inspect_dataset.py     # columns, splits, samples, distributions
│   ├── launch_hf_job.py       # `hf jobs run` wrapper with pre-flight check
│   └── eval_lm.py             # lm-eval-harness convenience wrapper
└── .claude/
    ├── settings.json          # allow-list of safe Bash invocations
    └── skills/
        ├── research-recipe/SKILL.md
        ├── inspect-dataset/SKILL.md
        ├── train-sft/SKILL.md
        ├── train-dpo/SKILL.md
        ├── launch-hf-job/SKILL.md
        ├── eval-model/SKILL.md
        └── env-bootstrap/SKILL.md
```

## Provenance

The agent rules and pre-flight discipline are adapted from `smolagents/ml-intern`'s
`agent/prompts/system_prompt_v3.yaml` (Apache-2.0). The local stack and skill
layout are original.
