# Distillery 🤖

A production-oriented agentic AI system for generating, critiquing, and evaluating LLM fine-tuning datasets.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
[![tests](https://github.com/AadiPathak23/distillery/actions/workflows/tests.yml/badge.svg)](https://github.com/AadiPathak23/distillery/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

Fine-tuning LLMs requires high-quality training data. Building that data by hand is slow, inconsistent, and hard to evaluate. Distillery automates the entire dataset engineering workflow through a multi-stage agentic pipeline.

The system follows a **Planner, Generator, Critic, Evaluator** architecture. Each stage operates as an independent agent with a well-defined responsibility. The Critic enforces strict dataset contracts (validated with pytest-style checks) and automatically rejects items that fail quality thresholds. Rejected slots are backfilled through an automated refill loop until the target count is met or retry limits are exhausted.

Every run produces a `debug.json` telemetry file that records generation counts, rejection reasons, and sample rejected items, making it straightforward to diagnose pipeline failures without guesswork.

The system supports local LLM inference via Ollama, cloud providers such as OpenAI, and a deterministic mock backend for testing. A Streamlit-based web UI is included for interactive configuration and result exploration.

---

## Architecture

```
User Prompt
    |
    v
Planner --> Dataset Generator --> Critic --> Evaluator
   |               |                |             |
   v               v                v             v
action_plan.md  dataset.json   (filter/refill)  evaluation.json
```

**Planner** -- Analyzes the user prompt and generates a structured action plan document covering target model analysis, dataset design rationale, risk assessment, and an implementation roadmap.

**Dataset Generator** -- Produces Q&A pairs using LLM inference (or template fallback). Generation is intent-driven: each dataset type is broken into distinct intents, and items are distributed across those intents to maximize diversity.

**Critic** -- Reviews every generated item against dataset-specific contracts. For example, `testcase_generation` items must contain a fenced Python code block with a `def test_` function, at least two assertions, and a pytest feature. An optional execution gate goes further, statically analyzing and *running* self-contained tests to reject ones that fail (see [How it verifies itself](#how-it-verifies-itself)). Items that fail are rejected, and the generator is invoked again to fill the gaps.

**Evaluator** -- Scores the final dataset on lexical diversity (TF-IDF, n-gram overlap), structural variety (question types, length distribution), conceptual coverage (LLM-assisted semantic analysis), and **correctness** (an LLM-judge rating faithfulness and usefulness). Produces an overall rating, health metrics, and actionable feedback.

---

## Key Features

- **Agentic multi-stage pipeline** -- Each stage (plan, generate, critique, evaluate) runs as a discrete agent with clear inputs and outputs.
- **Strict dataset contracts** -- Enforced per dataset type. `testcase_generation` requires valid pytest code; `bugfixing` requires diagnostic structure. Violations cause automatic rejection.
- **Execution-based correctness gate** -- For test-case datasets, generated pytest is statically analyzed (undefined names, parametrize/signature mismatches) and *actually executed* in a sandboxed subprocess. Tests that fail are rejected and regenerated — not shipped.
- **Correctness scoring** -- Beyond diversity, an LLM-judge rates each pair on faithfulness and usefulness; that score is folded into the overall rating so the headline number reflects correctness, not just variety.
- **Bring-your-own-key** -- Paste an OpenAI / Groq / OpenAI-compatible key in the UI (session-only, never written to disk), or run fully offline with Ollama.
- **Intent-based generation** -- The generator proposes intents per type (e.g., `error_diagnosis`, `fix_implementation`, `prevention`) and distributes items across them for coverage.
- **Refill loop** -- Rejected items trigger re-generation. The loop runs up to a configurable number of iterations to meet the requested item count.
- **Debug telemetry** -- Every run writes `debug.json` with pre-critique counts, rejection reasons, sample rejected items, refill iterations, and verification-coverage stats.
- **Multiple export formats** -- Outputs Q&A, Alpaca-style instruction, OpenAI chat JSONL, and a curated golden set for evaluation.

---

## How it verifies itself

Most synthetic-data tools trust the model's output. Distillery doesn't. For `testcase_generation`,
every generated test runs through a layered gate before it can enter the dataset:

1. **Syntax** — the code must parse.
2. **Parametrize/signature consistency** — `@pytest.mark.parametrize` argument names must exist in the test signature.
3. **Static analysis** — pyflakes rejects undefined names and missing imports.
4. **Execution** — self-contained tests are run under `pytest` in a sandboxed subprocess; failures are rejected.

Tests that can't be run fairly (they depend on an external module) are *skipped, not passed* — the pipeline never claims to have verified something it didn't. Each run reports a coverage line such as *"executed 9/10 generated tests, 8 passed, 1 rejected as broken."*

**Honest by design:** the resolved provider/model is printed before generation (a loud warning if it falls back to `mock`), and a run that filters everything raises an error with counts and reasons rather than silently writing an empty dataset.

> **Known limitation:** the execution gate only validates self-contained tests, and the correctness score is an LLM-judge (not ground truth). These are surfaced in telemetry rather than hidden.

<!-- SCREENSHOTS: add UI screenshots / a run GIF here before publishing -->

---

## Quickstart

### Install

```bash
git clone https://github.com/AadiPathak23/distillery.git
cd distillery
pip install -e ".[dev]"
```

### Run in Mock Mode

No API key required. Uses a deterministic template backend for testing the pipeline end to end.

```bash
python -m distillery
```

### Run Streamlit UI

```bash
streamlit run src/distillery/ui/app.py
```

### Run with Local LLM (Ollama)

```bash
# 1. Install Ollama from https://ollama.com/download
# 2. Pull a model
ollama pull qwen2.5-coder

# 3. Set the provider and run
set LLM_PROVIDER=ollama
streamlit run src/distillery/ui/app.py
```

On Linux/macOS, replace `set` with `export`.

---

## Example Use Case

Suppose you are fine-tuning a code LLM to write pytest test cases. You would:

1. Launch the Streamlit UI or CLI.
2. Enter a prompt such as: *"A Python testing assistant that generates pytest test cases for common utility functions."*
3. Select `testcase_generation` as the dataset type and set the count to 50.
4. Set the target model family to **Code LLM** and difficulty to **medium**.
5. Run the agent.

The pipeline generates 50 Q&A pairs where each answer contains a fenced Python code block with a valid `def test_` function, at least two `assert` statements, and a pytest feature such as `parametrize`, `raises`, or a fixture. Items that violate these contracts are automatically rejected and regenerated. The final output includes `dataset_chat.jsonl` ready for OpenAI-format fine-tuning and a `golden_set.jsonl` with the highest-quality items for use as a validation set.

---

## Project Structure

```
distillery/
├── src/distillery/
│   ├── agent.py              # Pipeline orchestration
│   ├── planner.py            # Action plan generation
│   ├── dataset_generator.py  # Intent-driven Q&A generation
│   ├── critic.py             # Contract enforcement and filtering
│   ├── evaluator.py          # Multi-metric quality scoring
│   ├── exporter.py           # JSONL export and golden set sampling
│   ├── schemas.py            # Pydantic data models
│   ├── cli.py                # Interactive CLI
│   ├── llm/
│   │   ├── base.py           # Abstract LLM interface
│   │   ├── openai.py         # OpenAI client
│   │   ├── ollama.py         # Ollama client
│   │   └── mock.py           # Deterministic mock client
│   ├── ui/
│   │   └── app.py            # Streamlit web interface
│   └── memory/
│       ├── store.py          # Abstract memory interface
│       ├── redis_store.py    # Redis backend
│       └── local_store.py    # JSON file backend
├── tests/
│   ├── test_evaluator.py
│   ├── test_generator.py
│   ├── test_critic.py
│   ├── test_exporter.py
│   └── test_agent_pipeline.py
├── artifacts/                # Generated outputs per run
├── pyproject.toml
└── README.md
```

---

## Deploy

The Streamlit app deploys to **[Streamlit Community Cloud](https://streamlit.io/cloud)** with no extra
config — it installs directly from `pyproject.toml`:

1. Push this repo to GitHub.
2. On Streamlit Community Cloud, create an app pointing at `src/distillery/ui/app.py`.
3. In the app, pick the **OpenAI-compatible** provider and paste your own API key (OpenAI or Groq).
   The key lives only in your session and is never written to disk or the repo.

Notes:
- The hosted app uses **bring-your-own-key** — there is no shared/server key to manage.
- **Ollama is local-only** (it needs a model server on your machine), so the local model path is for
  cloning and running the repo yourself, not the hosted demo.
- **Vercel is not suitable** for this app — it targets serverless/Next.js, whereas Streamlit needs a
  long-running server. Use Streamlit Community Cloud, Render, or Railway.

---

## Roadmap

- GitHub repository ingestion for context-aware dataset generation
- Embedding-based semantic deduplication to replace lexical similarity
- Multi-turn conversation dataset support
- Direct export to HuggingFace dataset format

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for the full text.
