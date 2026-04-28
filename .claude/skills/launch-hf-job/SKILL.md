---
name: launch-hf-job
description: Submit a training/inference script to Hugging Face Jobs (`hf jobs run`). Triggered when the user wants to run training in the cloud, scale beyond local hardware, or kick off a multi-hour fine-tune. Enforces pre-flight: hub_model_id required, ≥2h timeout for training, single-job validation before batch.
---

# launch-hf-job — Hugging Face Jobs submission

Wrapper: `scripts/launch_hf_job.py`. Refuses to launch unless the pre-flight
discipline is satisfied.

## Required, every time

- `--hub-model-id you/repo` — job storage is ephemeral. Without push_to_hub the
  trained model is permanently lost.
- `--timeout ≥2h` for training. Default 4h; scale to model size.
- Reference script must already have been smoke-tested locally (or in a tiny
  GPU sandbox) on the smallest sane subset.

## Hardware sizing

| Params | Flavor | Min timeout |
|---|---|---|
| 1–3B | `a10g-largex2` | 2h |
| 7–13B | `a100-large` | 6h |
| 30B+ | `l40sx4` or `a100x4` | 12h |
| 70B+ | `a100x8` | 24h |

`a10g-small` and `a10g-large` have the same 24 GB GPU — only CPU/RAM differ.
Pick `large` for the extra system RAM.

## Batch / ablation jobs

Submit ONE job first. Read the logs (`hf jobs logs <id>`) until you see the
loss come down for at least 10 logging steps. THEN submit the rest.

Never submit a sweep all at once — one bug kills every run identically.

## Common failure modes

| Symptom | Fix |
|---|---|
| Job killed at 30 minutes | You forgot `--timeout`. |
| Trained model gone | `push_to_hub=False` or `hub_model_id` missing. |
| `flash-attn` ImportError | Add it to `--extra-dep flash-attn`. |
| `KeyError: 'chosen'` | Audit dataset; you sent SFT data to DPO. |
| Gated dataset prompt | `HF_TOKEN` lacks scope; regenerate with read access. |

## Example

```bash
python scripts/launch_hf_job.py \
    --script scripts/train_sft.py \
    --config configs/my-run.yaml \
    --hardware a10g-largex2 \
    --timeout 4h \
    --hub-model-id my-user/smollm2-360m-capybara-sft
```

## OOM recovery in HF Jobs

Same playbook as local OOM (see `train-sft` skill): halve micro-batch, double
grad-accum, `gradient_checkpointing=True`, `bf16=True`, `adamw_8bit`, then
upgrade flavor. Never silently switch SFT → LoRA or drop `max_seq_length`.
