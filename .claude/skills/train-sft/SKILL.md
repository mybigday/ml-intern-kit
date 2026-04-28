---
name: train-sft
description: Supervised fine-tune a causal LM with TRL `SFTTrainer`. Triggered when the user wants to fine-tune / SFT / instruct-tune / chat-tune a model on conversational, prompt-completion, or text-formatted data. Enforces the literature-first → audit → smoke-test → scale workflow.
---

# train-sft — SFT recipe

Use the working template at `scripts/train_sft.py`. Do **not** write the
trainer from scratch — TRL renames things every release.

## Sequence (no shortcuts)

1. **Research** — fire `research-recipe` if you don't already have a paper-
   backed recipe. Capture: dataset, method, hyperparameters, reference URL.
2. **Audit** — `python scripts/inspect_dataset.py <dataset>`. Confirm:
   - Columns match SFT format (`messages`, `text`, or `prompt`+`completion`).
   - p95 sequence length informs `max_seq_length`.
3. **Config** — copy `configs/sft_default.yaml` to `configs/<run-name>.yaml`
   and override `model.name_or_path`, `dataset.name`, hyperparameters, and
   (when ready to ship) `train.push_to_hub: true` + `train.hub_model_id`.
4. **Smoke test** locally:
   ```bash
   python scripts/train_sft.py --config configs/<run-name>.yaml \
       --max-steps 20 --max-samples 256
   ```
   Loss must come down. If `push_to_hub` is on, the smoke test creates the
   repo — that's fine.
5. **Pre-flight block** — print this to the user before scaling:
   ```
   - Reference implementation: <paper / GH URL>
   - Dataset format verified : <yes — columns: …>
   - Model verified          : <repo_id, arch, tokenizer>
   - push_to_hub             : True, hub_model_id=<…>
   - Timeout                 : <≥2h>
   - Monitoring              : Trackio dashboard URL
   ```
6. **Scale** — either:
   - Local: `accelerate launch scripts/train_sft.py --config configs/<run>.yaml`
   - HF Jobs: `python scripts/launch_hf_job.py --script scripts/train_sft.py
     --config configs/<run>.yaml --hardware <flavor> --timeout <≥2h>
     --hub-model-id <user/repo>`

## Logging discipline (don't change)

In every TrainingArguments / SFTConfig:

```python
disable_tqdm=True
logging_strategy="steps"
logging_first_step=True
logging_steps=10
report_to=["trackio"]   # or ["wandb"] if WANDB_API_KEY is set
```

Loss prints as plain text lines you can `grep`.

## OOM

Apply in this order, **never** changing the user's request:

1. Halve `per_device_train_batch_size`, double `gradient_accumulation_steps`.
2. `gradient_checkpointing=True`.
3. `optim="adamw_8bit"` (bitsandbytes).
4. `bf16=True` (Ampere+).
5. Larger GPU tier.

Do **not** silently switch full SFT to LoRA, drop `max_seq_length`, or remove
monitoring. Those change what the user gets.
