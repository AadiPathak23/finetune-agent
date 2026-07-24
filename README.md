# Distillery ⚗️

**An agentic system that generates LLM fine-tuning datasets — and verifies its own output by
executing it, so you never ship training data you haven't actually checked.**

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
[![tests](https://github.com/AadiPathak23/distillery/actions/workflows/tests.yml/badge.svg)](https://github.com/AadiPathak23/distillery/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The problem

A fine-tuned model is only ever as good as the data it was trained on. Yet producing that data is
the single most under-tooled bottleneck in the whole fine-tuning workflow:

- **It's manual and slow.** Data/ML engineers hand-author thousands of instruction–response pairs,
  or hand-review what a model generated. It doesn't scale.
- **It's inconsistent.** Quality drifts across a dataset — some examples are sharp, others are
  shallow duplicates — and there's rarely an objective way to measure it.
- **Worst of all, it's unverified.** Existing synthetic-data tools *trust the model's output*: they
  ask an LLM for training examples and ship whatever comes back. For code datasets this is dangerous
  — a model will cheerfully write a "test" that looks correct and **fails the moment you run it**.
  That broken example goes straight into your training set and silently degrades the fine-tune.

So the fine-tuning engineering function is stuck choosing between two bad options: burn weeks
hand-reviewing examples, or ship unverified data and hope. **Distillery removes that trade-off.**

## The solution

Distillery is a multi-agent pipeline that **plans, generates, critiques, and evaluates** fine-tuning
datasets end to end — and for code datasets it doesn't trust the model, it **runs the generated tests
and rejects the ones that fail.** It refuses to ship broken or empty data, and every run emits a full
telemetry trail so you can see exactly what was produced, what was rejected, and why.

The result: instead of a pile of unverified Q&A pairs, you get a dataset where **the code examples
have provably executed and passed**, scored on diversity *and* correctness, exported in the formats
fine-tuning frameworks expect — produced in minutes, not weeks.

---

## How it verifies itself (the core idea)

Most synthetic-data tools trust the model. Distillery doesn't. For `testcase_generation`, every
generated test runs through a layered gate before it's allowed into the dataset:

1. **Syntax** — the code must parse.
2. **Parametrize / signature consistency** — `@pytest.mark.parametrize` argument names must exist in
   the test signature.
3. **Static analysis** — `pyflakes` rejects undefined names and missing imports.
4. **Execution** — self-contained tests are actually run under `pytest` in a sandboxed subprocess
   (with a timeout); tests that fail are rejected and regenerated, **not shipped**.

Tests that can't be run fairly (they depend on an external module) are *skipped, not passed* — the
pipeline never claims to have verified something it didn't. Every run reports a coverage line such as
*"executed 9/10 generated tests, 8 passed, 1 rejected as broken."*

**Honest by design:** the resolved provider/model is printed before generation (with a loud warning
if it falls back to `mock`), and a run that filters everything **raises an error with counts and
reasons** rather than silently writing an empty dataset.

> **Known limitations (surfaced, not hidden):** the execution gate only fully validates
> self-contained tests, and the correctness score is an LLM-judge rating (not ground truth). Both are
> reported in telemetry.

<!-- SCREENSHOTS: add UI screenshots (empty-state dashboard + a run's "Self-verified" banner + the evaluation panel) here before publishing -->

---

## Architecture

```
                        User prompt
                             |
                             v
   Planner  ──▶  Dataset Generator  ──▶  Critic  ──▶  Evaluator
      |                  |                  |             |
      v                  v                  v             v
 action_plan.md     dataset.json    (gate/filter/refill)  evaluation.json
                                          │
                                   execution gate ⇄ refill loop
```

**Planner** — Turns the prompt into a structured action-plan document: target-model analysis, dataset
design rationale, risk assessment, and an implementation roadmap.

**Dataset Generator** — Produces Q&A pairs via LLM inference (with a template fallback). Generation is
*intent-driven*: each dataset type is split into distinct intents (e.g. `error_diagnosis`,
`fix_implementation`, `prevention`) and items are distributed across them to maximize diversity.

**Critic** — Reviews every item against dataset-specific contracts, and for test-case data runs the
**execution gate** above. Rejected slots feed an automated **refill loop** that regenerates until the
target count is met or retry limits are exhausted.

**Evaluator** — Scores the final dataset on lexical diversity (TF-IDF, n-gram overlap), structural
variety, conceptual coverage (LLM-assisted), and **correctness** (an LLM-judge rating faithfulness +
usefulness), folded into a single overall rating plus health metrics and actionable feedback.

### Two-model split (strong generator, separate judge)

Distillery can run **generation and judging on different models**. Use a strong model
(e.g. `llama-3.3-70b-versatile`) to *write* the dataset, and a lighter one
(e.g. `llama-3.1-8b-instant`) to *critique and score* it. Besides matching each model to the job, on
providers with per-model rate limits (like Groq) this puts the judging calls on a **separate rate-limit
bucket**, so the correctness judge isn't starved by the generation workload. Falls back to a single
model when no judge model is selected.

---

## Key features

- **Agentic multi-stage pipeline** — plan → generate → critique → evaluate, each a discrete agent.
- **Execution-based correctness gate** — generated pytest is statically analyzed *and actually
  executed* in a sandboxed subprocess; failing tests are rejected, not shipped.
- **Two-model split** — strong model for generation, lighter/separate model for judging (see above).
- **Correctness scoring** — an LLM-judge rates faithfulness + usefulness; folded into the overall
  rating so the headline number reflects correctness, not just variety.
- **Strict dataset contracts** — enforced per type; violations are auto-rejected.
- **Refill loop** — rejected items trigger bounded regeneration to hit the requested count.
- **Rate-limit resilience** — exponential backoff on transient 429/5xx, with a capped `Retry-After`
  so a provider's quota-reset signal can never hang a run.
- **Bring-your-own-key** — paste an OpenAI / Groq / OpenAI-compatible key in the UI (session-only,
  never written to disk; env keys are used silently and never rendered), or run offline with Ollama.
- **Professional web UI** — a Streamlit dashboard (sidebar control panel + results canvas) with a
  live "Self-verified" banner, score cards, and per-format downloads.
- **Debug telemetry** — every run writes `debug.json` (pre-critique counts, rejection reasons, sample
  rejections, refill iterations, verification-coverage stats).
- **Multiple export formats** — Q&A, Alpaca-style instruction, OpenAI chat JSONL, and a curated
  `golden_set.jsonl` for use as a validation set.

---

## Quickstart

### Install

```bash
git clone https://github.com/AadiPathak23/distillery.git
cd distillery
pip install -e ".[dev]"
```

### Web UI (recommended)

```bash
streamlit run src/distillery/ui/app.py
```

Pick a provider in the sidebar (**Groq**, **OpenAI-compatible**, **Ollama**, or **Mock**), configure
the run, and hit **Run Agent**.

### Mock mode (no API key)

Uses a deterministic template backend to exercise the whole pipeline end to end:

```bash
python -m distillery
```

### With a real model

Copy `.env.example` to `.env` and paste a key. For **Groq** (free, OpenAI-compatible) the file is
already set up — you only paste the key:

```dotenv
LLM_PROVIDER=openai
GROQ_API_KEY=gsk_your_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.3-70b-versatile
```

For **Ollama** (local): install from https://ollama.com/download, `ollama pull qwen2.5-coder`, set
`LLM_PROVIDER=ollama`, and run the UI.

---

## Example use case

You're fine-tuning a code LLM to write pytest test cases. In the UI:

1. Prompt: *"A Python testing assistant that generates pytest test cases for common utility functions."*
2. Dataset type: `testcase_generation`, count 50, target **Code LLM**, difficulty **medium**.
3. Enable the **correctness gate**, and (optionally) set a lighter **judge model**.
4. Run.

Distillery generates 50 Q&A pairs where each answer is a self-contained, runnable pytest block. Items
that don't parse, lack a `def test_`, have too few assertions, or **fail when executed** are rejected
and regenerated. You get `dataset_chat.jsonl` ready for OpenAI-format fine-tuning and a
`golden_set.jsonl` of the highest-quality items for validation — plus a banner telling you exactly how
many tests were executed and passed.

---

## Project structure

```
distillery/
├── src/distillery/
│   ├── agent.py              # Pipeline orchestration (+ optional two-model split)
│   ├── planner.py            # Action-plan generation
│   ├── dataset_generator.py  # Intent-driven Q&A generation (+ retry/backoff)
│   ├── critic.py             # Contract enforcement and filtering
│   ├── critic_execution.py   # Execution-based correctness gate (sandboxed pytest)
│   ├── evaluator.py          # Multi-metric quality scoring (+ correctness judge)
│   ├── exporter.py           # JSONL export and golden-set sampling
│   ├── schemas.py            # Pydantic data models
│   ├── cli.py                # Interactive CLI
│   ├── llm/                  # openai / ollama / mock clients + factory
│   ├── ui/app.py             # Streamlit dashboard
│   └── memory/               # store abstraction (Redis / local JSON)
├── tests/                    # pytest suite (136 tests)
├── artifacts/                # Generated outputs per run
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Deploy

The Streamlit app deploys to **[Streamlit Community Cloud](https://streamlit.io/cloud)** with no extra
config — it installs directly from `pyproject.toml`:

1. Push this repo to GitHub.
2. On Streamlit Community Cloud, create an app pointing at `src/distillery/ui/app.py`.
3. In the app, pick a provider and paste your own API key (OpenAI or Groq). The key lives only in
   your session and is never written to disk or the repo.

Notes:
- The hosted app uses **bring-your-own-key** — there is no shared/server key to manage.
- **Ollama is local-only** (it needs a model server on your machine).
- **Vercel is not suitable** — it targets serverless/Next.js, whereas Streamlit needs a long-running
  server. Use Streamlit Community Cloud, Render, or Railway.

---

## Roadmap

- GitHub repository ingestion for context-aware dataset generation
- Embedding-based semantic deduplication to replace lexical similarity
- Multi-turn conversation dataset support
- Direct export to HuggingFace dataset format

---

## License

MIT — see [LICENSE](LICENSE).
