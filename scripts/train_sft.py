"""Supervised fine-tune with TRL `SFTTrainer`.

Working template — adapted from current TRL examples. The intent is that the
agent reads this file before writing trainers, rather than guessing arg names.

Quick smoke test:
    python scripts/train_sft.py --config configs/sft_default.yaml \
        --max-steps 20 --max-samples 256

Full run:
    accelerate launch scripts/train_sft.py --config configs/sft_default.yaml \
        --hub-model-id you/your-model

Submit to HF Jobs:
    python scripts/launch_hf_job.py --script scripts/train_sft.py \
        --config configs/sft_default.yaml --hardware a10g-largex2 --timeout 4h
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


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_report_to(report_to: list[str]) -> list[str]:
    """If wandb is requested but no key is set, fall back to trackio."""
    out = list(report_to)
    if "wandb" in out and not os.environ.get("WANDB_API_KEY"):
        out = [r for r in out if r != "wandb"] or ["trackio"]
    if "trackio" in out:
        try:
            import trackio  # noqa: F401
        except ImportError:
            out = [r for r in out if r != "trackio"] or ["tensorboard"]
    return out


def _platform_overrides(model_cfg: dict, train_cfg: dict) -> tuple[dict, dict]:
    """Adjust dtype / attn_impl / optim / precision flags for the active backend.

    - CUDA / ROCm Ampere+: keep user choice (typically bf16 + sdpa or flash-attn-2).
    - macOS MPS:           force fp16 (bf16 is gimped on MPS), drop flash-attn-2,
                           force adamw_torch (bitsandbytes unavailable).
    - CPU:                 force fp32, drop flash-attn-2, adamw_torch.
    """
    import torch

    has_cuda = torch.cuda.is_available()
    is_rocm = bool(getattr(torch.version, "hip", None)) and has_cuda
    has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

    m, t = dict(model_cfg), dict(train_cfg)

    if has_cuda and not is_rocm:
        return m, t  # pristine path

    if is_rocm:
        # ROCm: bf16 OK on MI200+; flash-attn upstream is CUDA-only.
        if m.get("attn_implementation") == "flash_attention_2":
            print("[train_sft] ROCm: flash_attention_2 unsupported, falling back to sdpa")
            m["attn_implementation"] = "sdpa"
        if t.get("optim") == "adamw_8bit":
            print("[train_sft] ROCm: bitsandbytes adamw_8bit unsupported, using adamw_torch")
            t["optim"] = "adamw_torch"
        return m, t

    if has_mps:
        if m.get("torch_dtype", "bfloat16") == "bfloat16":
            print("[train_sft] MPS: bfloat16 is poorly supported, switching to float16")
            m["torch_dtype"] = "float16"
        if m.get("attn_implementation") == "flash_attention_2":
            print("[train_sft] MPS: flash_attention_2 unsupported, falling back to sdpa")
            m["attn_implementation"] = "sdpa"
        t["bf16"] = False
        t["fp16"] = True
        if t.get("optim") in ("adamw_8bit", "paged_adamw_32bit"):
            print("[train_sft] MPS: bitsandbytes optimizers unsupported, using adamw_torch")
            t["optim"] = "adamw_torch"
        return m, t

    # CPU
    print("[train_sft] CPU-only: forcing float32 + adamw_torch + sdpa attention")
    m["torch_dtype"] = "float32"
    m["attn_implementation"] = "sdpa"
    t["bf16"] = False
    t["fp16"] = False
    t["optim"] = "adamw_torch"
    return m, t


def main(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_cfg = cfg["model"]
    data_cfg = cfg["dataset"]
    peft_cfg = cfg.get("peft") or {}
    train_cfg = cfg["train"]
    train_cfg["report_to"] = _resolve_report_to(train_cfg.get("report_to", ["trackio"]))
    model_cfg, train_cfg = _platform_overrides(model_cfg, train_cfg)

    if args.hub_model_id:
        train_cfg["hub_model_id"] = args.hub_model_id
        train_cfg["push_to_hub"] = True

    # Load tokenizer + model
    print(f"[train_sft] loading model: {model_cfg['name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        torch_dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )

    # Optional LoRA
    peft_config = None
    if peft_cfg.get("adapter") == "lora":
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=peft_cfg.get("r", 16),
            lora_alpha=peft_cfg.get("lora_alpha", 32),
            lora_dropout=peft_cfg.get("lora_dropout", 0.05),
            bias=peft_cfg.get("bias", "none"),
            target_modules=peft_cfg.get("target_modules"),
            task_type="CAUSAL_LM",
        )

    # Dataset
    print(f"[train_sft] loading dataset: {data_cfg['name']} split={data_cfg.get('split', 'train')}")
    ds = load_dataset(data_cfg["name"], split=data_cfg.get("split", "train"))
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"[train_sft] truncated dataset to {len(ds)} samples")

    # SFTConfig — pass training kwargs straight through. TRL ≥ 0.12 inherits from TrainingArguments.
    sft_kwargs = dict(train_cfg)
    if args.max_steps:
        sft_kwargs["max_steps"] = args.max_steps
        sft_kwargs.pop("num_train_epochs", None)
    sft_kwargs["max_seq_length"] = data_cfg.get("max_seq_length", 2048)
    sft_kwargs["dataset_text_field"] = data_cfg.get("text_field", "text")

    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("[train_sft] starting training")
    trainer.train()

    print("[train_sft] saving final model to:", sft_config.output_dir)
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)

    if sft_config.push_to_hub and sft_config.hub_model_id:
        print(f"[train_sft] pushing to hub: https://huggingface.co/{sft_config.hub_model_id}")
        trainer.push_to_hub()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="YAML recipe (see configs/sft_default.yaml)")
    p.add_argument("--max-steps", type=int, default=None, help="Cap training steps (smoke tests).")
    p.add_argument("--max-samples", type=int, default=None, help="Cap dataset size (smoke tests).")
    p.add_argument("--hub-model-id", type=str, default=None, help="Override push_to_hub repo id.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    main(cfg, args)
