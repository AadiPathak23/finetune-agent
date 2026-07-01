# Project Memory — Finetune Agent

> Working log of changes made to this codebase, so a fresh session (human or AI)
> can pick up without re-deriving context. Newest context at top.
> Last updated: 2026-06-30.

## What this project is

An agentic tool that generates fine-tuning datasets (Q&A pairs) for code LLMs.
Pipeline lives in `src/finetune_agent/agent.py` (`FinetuneAgent.run`):

`Planner → Generator → Critic → refill loop → filter → Evaluator → export`

Key modules:
- `dataset_generator.py` — `LLMDatasetGenerator` (LLM-backed) + `TemplateGenerator` (fallback).
- `critic.py` — `DatasetCritic`: rule-based + LLM critique, filters low-quality items.
- `evaluator.py` — `Evaluator`: scores datasets.
- `llm/` — provider clients: `openai.py`, `ollama.py`, `mock.py`; factory in `llm/__init__.py`.
- `schemas.py` — Pydantic models. `ui/app.py` — Streamlit UI. `cli.py` — CLI entry point.

## Environment / how to run with a real model

Config is loaded from a project-root `.env` automatically at startup (see below).
To use **Groq** (free, OpenAI-compatible), put this in `.env`:

```dotenv
LLM_PROVIDER=openai
GROQ_API_KEY=gsk_...real key...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-8b-instant
```

- Valid current Groq models (checked against Groq's live changelog): `llama-3.1-8b-instant`
  (small/fast), `llama-3.3-70b-versatile` (higher quality), `openai/gpt-oss-20b|120b`.
- `LLM_PROVIDER` options: `openai` | `ollama` | `mock`. **`mock` is the default** and
  produces canned/templated output (not a real model).
- Sanity check the resolved provider:
  ```
  python -c "import sys; sys.path.insert(0,'src'); import finetune_agent; from finetune_agent.llm import get_llm_client as g; c=g(); print(c.provider_name, getattr(c,'_model',''))"
  ```
  Must print `openai ...`, not `mock`.

## Changes made so far (chronological)

### 1. Diagnosis of weak output quality
Root causes identified (still partly open — see "Known issues / TODO"):
- All recent runs silently used the **mock** generator (`llm_provider: 'mock'` in
  `artifacts/*/dataset.json`) — i.e. canned content, not a real LLM.
- Several runs silently exported **0 items** (everything filtered by the Critic).
- The **Evaluator scores only diversity/length/balance, not correctness** — so weak
  content can still score "fine".
- The **Critic barely filters** in default (non-aggressive) mode, and its placeholder
  regex (`\[.*?\]`, `<.*?>`, `\.\.\.` in `critic.py:_check_answer_quality`) over-flags
  normal code (list literals, type hints, parametrize tables). In aggressive mode it
  would over-delete valid code.

### 2. Honesty changes — `agent.py` (`FinetuneAgent.run`)
- **Prints resolved provider + model** before generation:
  `Generation provider: <p> | model: <m>` (also stored in `debug.json` as
  `llm_provider` / `llm_model`).
- **Loud WARNING** when provider resolves to `mock`.
- **Raises `GenerationError`** instead of silently writing an empty dataset; the message
  reports generated count, rejected count, and top rejection reasons.

### 3. Groq support (OpenAI-compatible) — minimal, non-breaking
- `llm/openai.py`: `OpenAIClient` now accepts `GROQ_API_KEY` as a fallback key
  (`OPENAI_API_KEY` still takes precedence). Custom base URL already supported via
  `OPENAI_BASE_URL`.
- `llm/__init__.py`: factory's `openai` branch accepts `OPENAI_API_KEY or GROQ_API_KEY`.

### 4. Rate-limit resilience — `dataset_generator.py`
- Added **`_generate_json_with_retry`**: exponential backoff (2s→4s→8s), retries on
  HTTP **429** and **5xx**, honors `Retry-After` header, raises immediately on
  non-retryable errors (e.g. 400/401). The QA-batch call now goes through it.
- **Fixed a Pydantic crash** in `_generate_template_batch`: removed a dead first
  `GenerationRequest(...)` built from a hand-rolled fake constraints object that failed
  validation before the correct `UserConstraints`-based construction ran. The template
  fallback now actually works.

### 5. `.env` auto-loading
- `src/finetune_agent/__init__.py`: loads project-root `.env` via `python-dotenv` on
  import (covers CLI, `python -m`, Streamlit, tests). Shell vars override the file
  (`override=False`). Degrades silently if `python-dotenv` is missing.
- `pyproject.toml`: added `python-dotenv>=1.0.0`.
- Added **`.env.example`** documenting every env var (`LLM_PROVIDER`, `GROQ_API_KEY`,
  `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OLLAMA_HOST`, `OLLAMA_MODEL`,
  `REDIS_URL`).
- `.env` is already in `.gitignore` (real keys never committed).

## All env vars the code reads
`LLM_PROVIDER`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`,
`OLLAMA_HOST`, `OLLAMA_MODEL`, `REDIS_URL`.

## Known issues / TODO (not yet done)
- [ ] **Do a real Groq generation run** end-to-end (needs a real key in `.env`); only
      placeholder-key wiring tests have been run so far.
- [ ] **Evaluator measures diversity, not correctness** — consider an LLM-judge rubric
      scoring faithfulness/usefulness per pair and feeding it into `overall_rating`.
- [ ] **Template fallback can't satisfy the pytest contract** for `testcase_generation`,
      so a forced fallback there yields an empty dataset (now surfaced via the honest
      `GenerationError` rather than a crash).
- [ ] **Critic placeholder regex** over-flags code; aggressive mode would over-delete
      valid code. Needs tightening before aggressive filtering is usable on code.
- [ ] Retry/backoff is only on the QA-batch call. `generate_intents`, the Critic, and the
      Evaluator have their own try/except fallbacks but no backoff — could add later.

## Files touched
`pyproject.toml`, `src/finetune_agent/__init__.py`, `src/finetune_agent/agent.py`,
`src/finetune_agent/dataset_generator.py`, `src/finetune_agent/llm/__init__.py`,
`src/finetune_agent/llm/openai.py`, plus new `.env.example` and this `memory.md`.
