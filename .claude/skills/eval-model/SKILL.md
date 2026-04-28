---
name: eval-model
description: Evaluate a trained or downloaded language model with `lm-eval-harness` standard tasks (arc, hellaswag, gsm8k, mmlu, truthfulqa, ifeval, ...). Triggered when the user wants to benchmark, eval, or compare a model — pre- or post-training.
---

# eval-model — lm-eval-harness wrapper

Wrapper: `scripts/eval_lm.py`. Requires `pip install 'lm-eval>=0.4.5'`
(or `bash bootstrap.sh --eval`).

## Quick eval

```bash
python scripts/eval_lm.py \
    --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --tasks arc_easy,hellaswag,piqa \
    --num-fewshot 0 \
    --batch-size auto
```

For local checkpoints, pass the directory (`outputs/sft-default`) or any Hub
repo id. `lm-eval` handles loading.

## Task picks by training goal

| Goal | Tasks |
|---|---|
| Instruct-quality smoke | `arc_easy,hellaswag,piqa,winogrande` |
| Math reasoning | `gsm8k,minerva_math` |
| Knowledge | `mmlu` |
| Truthfulness | `truthfulqa_mc2` |
| Instruction following | `ifeval` |
| Code | `humaneval` (needs sandbox) |

## Reporting

After the run, surface:

- **Task → score → baseline** comparison. Use the published numbers from the
  paper that motivated the recipe (the `research-recipe` output should include
  these).
- The model URL on the Hub.
- The Trackio / wandb run URL for training, if applicable.

Do not just dump the JSON. Pick the 3-5 numbers the user actually cares about
and put them in a small table.
