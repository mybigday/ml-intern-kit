---
name: research-recipe
description: Run a literature-first crawl before writing ANY ML training/fine-tuning/inference code. Spawns an Explore sub-agent that mines papers, citation graphs, methodology sections, and matched HF datasets to produce a ranked list of training recipes attributed to specific published results. Triggered when the user asks to fine-tune, train, or improve a model, or when the user names a task/benchmark and you need a recipe before coding.
---

# research-recipe — literature-first training recipe

Your knowledge of TRL/Transformers/PEFT APIs is outdated. Your knowledge of which
dataset+method combo produces the best result on benchmark Z is even more
outdated. **Always crawl the literature first.**

## When to fire

- The user asks to fine-tune / train / improve a model.
- The user names a task or benchmark ("RAG with citations", "code completion on
  HumanEval", "math reasoning on MATH").
- Before writing a trainer config from scratch.

Skip only for trivial non-code operations.

## How to fire

Spawn a sub-agent so paper text never enters the main context:

```
Agent(
  description="Literature crawl for <task>",
  subagent_type="Explore",
  prompt="""
Literature crawl for <TASK>. Start from <ANCHOR PAPER OR TOPIC>.

1. Find anchor papers: search arxiv + HF papers for the task. Identify 2-3
   landmark papers (high citations, recent, or both). Use WebSearch and
   `WebFetch https://huggingface.co/papers?q=<task>`.

2. Crawl citation graph DOWNSTREAM: papers that cite the anchors and improved
   on them. Hit Semantic Scholar:
   `WebFetch https://api.semanticscholar.org/graph/v1/paper/arXiv:<id>/citations?fields=title,year,citationCount,abstract&limit=30`.

3. Read methodology sections (sections 3, 4, 5 — Methodology / Experiments /
   Results) of the 3-5 most promising recent papers. Use the arxiv HTML view:
   `WebFetch https://arxiv.org/html/<id>v1` or the PDF via abs page.

4. For each paper, extract:
   - DATASET: name, size, source, HF Hub repo if any, format (messages /
     prompt+completion / prompt+chosen+rejected).
   - METHOD: optimizer, learning rate, schedule, epochs, effective batch size,
     sequence length, key tricks (packing, FlashAttention, curriculum, …).
   - RESULT: exact numbers on a named benchmark.

5. Find ONE working code example using current TRL/Transformers APIs. Use
   `gh search code 'SFTTrainer path:examples extension:py' --limit 10` then
   `gh api repos/<owner>/<repo>/contents/<path>` to read it.

6. Cross-check the top dataset(s) on HF Hub via
   `WebFetch https://huggingface.co/api/datasets/<repo_id>` for downloads,
   recent activity, and (optionally) audit columns by suggesting
   `python scripts/inspect_dataset.py <repo_id>` to the main agent.

OUTPUT (500-1500 words):
- Ranked list of recipes. For each:
  - Paper: title, arxiv_id, date, venue
  - Result: exact numbers
  - Dataset(s): name, size, HF availability, format verified (yes/no)
  - Method: training approach + hyperparameters
  - "What made it work": specific insight
- SOTA landscape (one paragraph).
- Code anchor: one URL/path of a current working example.
- Risks: anything you saw in the literature that commonly breaks training
  (OOM, format mismatch, deprecated args).
"""
)
```

## After the sub-agent returns

Translate the top recipe into:

1. A modified `configs/sft_default.yaml` (or new `configs/<recipe>.yaml`).
2. A `scripts/inspect_dataset.py <dataset>` call to confirm format.
3. The pre-flight block from `AGENTS.md` ("Reference implementation: …").

Do not write trainer code from memory. Read the example URL the sub-agent
returned, then adapt the matching template under `scripts/`.
