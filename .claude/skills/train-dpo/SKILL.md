---
name: train-dpo
description: Direct Preference Optimization (DPO) fine-tune with TRL `DPOTrainer`. Triggered when the user wants to align a model on preferences / pairwise comparisons / chosen-vs-rejected data, or improve an existing SFT checkpoint with a preference dataset.
---

# train-dpo — preference optimization

Template: `scripts/train_dpo.py`. Required dataset columns: `prompt`,
`chosen`, `rejected`.

## Sequence

1. **Audit dataset** — `python scripts/inspect_dataset.py <dataset>`. Verify
   the three required columns exist and aren't empty / duplicated.
2. **Pick the SFT base** — DPO requires a model that has already been instruct-
   tuned. If the user gives a base model that hasn't been SFT'd, surface this
   and propose either: (a) SFT first on a small prompt-completion dataset, or
   (b) start from a published instruct checkpoint (e.g. SmolLM2-360M-Instruct).
3. **Config** — start from `configs/sft_default.yaml`; for DPO add/override:
   ```yaml
   train:
     beta: 0.1                  # KL coefficient — typical range 0.05 – 0.5
     max_length: 2048
     max_prompt_length: 1024
     learning_rate: 5.0e-7      # DPO is much more sensitive than SFT
     num_train_epochs: 1
   ```
4. **Smoke test** — same `--max-steps 20 --max-samples 256` discipline.
5. **Watch reward margin and KL** in Trackio. If `rewards/margin` stays at 0
   the dataset is uninformative. If KL blows up, lower the learning rate.
6. **Scale** locally with `accelerate` or via `scripts/launch_hf_job.py`.

DPO doubles GPU memory (policy + reference model). Size up one tier from what
you'd pick for SFT of the same model.
