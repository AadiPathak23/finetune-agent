# Distillery 🤖

A production-oriented agentic AI system for generating, critiquing, and evaluating LLM fine-tuning datasets.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Tests Passing](https://img.shields.io/badge/Tests-Passing-brightgreen)
![Status: Active](https://img.shields.io/badge/Status-Active-blue)

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

**Critic** -- Reviews every generated item against dataset-specific contracts. For example, `testcase_generation` items must contain a fenced Python code block with a `def test_` function, at least two assertions, and a pytest feature. Items that fail are rejected, and the generator is invoked again to fill the gaps.

**Evaluator** -- Scores the final dataset on lexical diversity (TF-IDF, n-gram overlap), structural variety (question types, length distribution), and conceptual coverage (LLM-assisted semantic analysis). Produces an overall rating, health metrics, and actionable feedback.

---

## Key Features

- **Agentic multi-stage pipeline** -- Each stage (plan, generate, critique, evaluate) runs as a discrete agent with clear inputs and outputs.
- **Strict dataset contracts** -- Enforced per dataset type. `testcase_generation` requires valid pytest code; `bugfixing` requires diagnostic structure. Violations cause automatic rejection.
- **Automated quality scoring** -- Uniqueness scores combine lexical (50%), structural (30%), and conceptual (20%) metrics into a single 0-100 rating.
- **Intent-based generation** -- The generator proposes intents per type (e.g., `error_diagnosis`, `fix_implementation`, `prevention`) and distributes items across them for coverage.
- **Refill loop** -- Rejected items trigger re-generation. The loop runs up to a configurable number of iterations to meet the requested item count.
- **Debug telemetry** -- Every run writes `debug.json` with pre-critique counts, rejection reasons, sample rejected items, and refill iteration counts.
- **Local LLM support** -- Full Ollama integration for running the entire pipeline against local models without external API calls.
- **Multiple export formats** -- Outputs Q&A, Alpaca-style instruction, OpenAI chat JSONL, and a curated golden set for evaluation.

---

## Quickstart

### Install

```bash
git clone <repository-url>
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

## Roadmap

- GitHub repository ingestion for context-aware dataset generation
- Embedding-based semantic deduplication to replace lexical similarity
- Multi-turn conversation dataset support
- Direct export to HuggingFace dataset format

---

## License

This project is licensed under the MIT License.
