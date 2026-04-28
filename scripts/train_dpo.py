"""DPO preference optimization with TRL `DPOTrainer`.

Dataset format must have columns: prompt, chosen, rejected.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def main(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    model_cfg = cfg["model"]
    data_cfg = cfg["dataset"]
    train_cfg = cfg["train"]

    if args.hub_model_id:
        train_cfg["hub_model_id"] = args.hub_model_id
        train_cfg["push_to_hub"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"], dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"], dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )

    ds = load_dataset(data_cfg["name"], split=data_cfg.get("split", "train"))
    required = {"prompt", "chosen", "rejected"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(
            f"DPO dataset {data_cfg['name']} is missing required columns: {missing}. "
            "Audit it with scripts/inspect_dataset.py first."
        )
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    dpo_kwargs = dict(train_cfg)
    if args.max_steps:
        dpo_kwargs["max_steps"] = args.max_steps
        dpo_kwargs.pop("num_train_epochs", None)
    dpo_kwargs.setdefault("beta", 0.1)
    dpo_kwargs.setdefault("max_length", data_cfg.get("max_seq_length", 2048))
    dpo_kwargs.setdefault("max_prompt_length", 1024)

    dpo_config = DPOConfig(**dpo_kwargs)

    trainer = DPOTrainer(
        model=model, ref_model=ref_model, args=dpo_config,
        train_dataset=ds, processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(dpo_config.output_dir)
    tokenizer.save_pretrained(dpo_config.output_dir)
    if dpo_config.push_to_hub and dpo_config.hub_model_id:
        trainer.push_to_hub()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--hub-model-id", type=str, default=None)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    main(cfg, args)
