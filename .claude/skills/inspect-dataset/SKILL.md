---
name: inspect-dataset
description: Audit a Hugging Face dataset before training to confirm splits, columns, format, sample rows, distributions, and duplicates. Triggered before any training/fine-tuning script runs, when a user mentions a new dataset, or when you hit a KeyError / format mismatch in a training job.
---

# inspect-dataset

Never train on a dataset you haven't audited. Looking at the data is the single
most reliable way to avoid format mismatches, KeyError-on-step-1 failures, and
silently-bad runs.

## When to fire

- Before any `train_sft.py` / `train_dpo.py` / `train_grpo.py` invocation.
- When the user says "use dataset X" — audit X immediately.
- When a training job fails with `KeyError`, `ValueError: column not found`, or
  bad samples.

## What to run

```bash
python scripts/inspect_dataset.py <repo_id>             # default audit
python scripts/inspect_dataset.py <repo_id> --split train --n 5
python scripts/inspect_dataset.py <repo_id> --config wikitext-2-raw-v1
```

The script prints schema, row counts per split, length statistics on text /
prompt / messages columns, duplicate counts, label distribution, and `n` sample
rows.

## What to verify against the chosen training method

| Method | Required columns |
|---|---|
| SFT  | `messages` (chat) **or** `text` **or** `prompt`+`completion` |
| DPO  | `prompt`, `chosen`, `rejected` |
| GRPO | `prompt` (+ a reward function) |

If the columns don't match the method, **stop**. Tell the user; don't silently
remap or substitute the dataset.

## Things to surface to the user

- Class imbalance, missing values, duplicate rows.
- Sequence-length p95 — drives the `max_seq_length` choice.
- Dataset license / gating.
- Any unexpected nesting (e.g. `messages` already wrapped in a list of dicts vs.
  flat text).
