# experiments/

Self-contained experiment folders. **Code is git-tracked, training outputs
are not.**

## Layout

Each experiment is a directory:

```
experiments/<YYYY-MM-DD-short-name>/
├── README.md         # tracked — hypothesis, recipe, results
├── config.yaml       # tracked — what train_sft.py / train_dpo.py / train_grpo.py reads
├── notes.md          # tracked — running journal, links, dead-ends
└── runs/             # auto-created on first run, IGNORED (trainer outputs, wandb, trackio)
```

The repo's `.gitignore` already swallows `runs/`, `outputs/`, `checkpoints/`,
`checkpoint-*/`, `wandb/`, `trackio/`, `*.safetensors`, `*.bin`, `*.pt`,
`*.gguf`, etc. at any depth — so you do not need a per-experiment `.gitignore`.
The `runs/` dir does not need to exist until the trainer creates it.

## Top-level ignored areas

| Path | Purpose |
|---|---|
| `data/`        | datasets you build or download by hand (HF datasets cache lives in `~/.cache/huggingface/` by default — point `HF_HOME=./hf_home` if you want it project-local) |
| `models/`      | model weights you download outside HF cache |
| `hf_home/`     | local override for `HF_HOME` if you want all HF state inside the repo |
| `outputs/`     | shared output dir for ad-hoc one-off runs (see `configs/sft_default.yaml`) |
| any `runs/`    | trainer working dirs anywhere in the tree |

## Start a new experiment

```bash
cp -r experiments/_template experiments/2026-04-28-my-recipe
$EDITOR experiments/2026-04-28-my-recipe/{README.md,config.yaml}

python scripts/train_sft.py --config experiments/2026-04-28-my-recipe/config.yaml
# trainer writes to experiments/2026-04-28-my-recipe/runs/  (ignored)
```

**`output_dir` resolution rule** in the train scripts:

| `output_dir` in YAML | Where it writes |
|---|---|
| `/abs/path/runs`     | exactly that absolute path |
| `./runs/sft`         | resolved against the config file's directory (so `experiments/<exp>/config.yaml` writes under `experiments/<exp>/runs/sft`) |
| `outputs/sft`        | left as-is, relative to CWD (the project root convention used by `configs/sft_default.yaml`) |

The template uses `./runs/sft` so each experiment is self-contained.

## Why this layout

- One `git log` shows recipe + result history alongside the code that
  produced it. The user's actual model bytes never touch the repo.
- Drop a folder, drop the experiment. No lingering `wandb/` or
  `checkpoint-12000/` to clean up by hand.
- The `_template/` skeleton is the canonical shape — copy it, don't invent.
