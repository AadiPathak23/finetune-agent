# Distillery — project guide for Claude

Distillery is an **agentic pipeline that generates LLM fine-tuning datasets and verifies its own
output by executing it** (for code datasets it runs the generated pytest and rejects failures). It's
a personal portfolio project headed for a **LinkedIn launch**. See `README.md` for the full picture
and `memory.md` for the running change log (newest at top).

## Repo facts
- GitHub: `AadiPathak23/distillery` (renamed from `finetune-agent`; `main` is the default branch).
- Python package: `src/distillery/`. Install: `pip install -e ".[dev]"`. Tests: `python -m pytest -q`
  (**136 passing**). Run the UI: `streamlit run src/distillery/ui/app.py`.
- Pipeline: `agent.py` orchestrates Planner → Generator → Critic (+ execution gate in
  `critic_execution.py`) → refill loop → Evaluator → export. LLM clients in `llm/`
  (`openai` handles OpenAI **and** Groq via base URL; plus `ollama`, `mock`).

## Working agreements (important)
- **Never add a `Co-Authored-By: Claude` trailer to commits.** Commits are authored solely under the
  owner (public showcase). Strip it if a hook adds it.
- **Do not weaken the correctness gate / critic contracts** to make output "pass." The gate rejecting
  broken tests (and the honest `GenerationError` on an empty dataset) is the core value, not a bug.
- The owner reviews UI changes visually. You can **screenshot the running Streamlit app** with
  headless Chrome via Playwright (`channel="chrome"`, no browser download) — a helper pattern was used
  this session; wait for `.distillery-hero` to render, then capture.

## Gotchas learned this session
- **Backend changes need a full server restart.** Streamlit hot-reloads `app.py` but keeps
  deeply-imported modules (e.g. `agent.py`) cached — a stale server throws confusing errors. On
  Windows, `pkill -f "streamlit run"` does **not** kill native `streamlit.exe`; use
  `taskkill //F //IM streamlit.exe`, confirm the port is free, then relaunch.
- **Groq free-tier limits are the current showcase blocker, not the code.** On a heavy day the daily
  budget throttles/truncates responses: generation quality drops (syntax fails → fewer survivors) and
  the correctness judge falls back to the neutral `70.0`. Real correctness is ~85 (proven in
  isolation). Fix = run on fresh quota or a paid tier / second provider for the judge.
- The two-model split (`FinetuneAgent(aux_llm_client=...)`) routes generation to the primary model and
  Critic+Evaluator to the aux model; falls back to primary if no aux.

## Current focus: DEPLOYMENT
The owner is starting this session to deploy. Key constraints:
- **Deploy target: Streamlit Community Cloud** (free; installs from `pyproject.toml`; point it at
  `src/distillery/ui/app.py`). Render or Railway also work. It's **bring-your-own-key** — no server
  key to manage; users paste their own in the sidebar.
- **Vercel does NOT work** for this app — it's serverless/Next.js, and Streamlit needs a long-running
  server (websocket). The execution gate also spawns pytest subprocesses, which serverless can't do.
  Don't try to force Vercel; steer to Streamlit Cloud / Render / Railway.
- Before/after deploy, the remaining launch assets are: a clean showcase run (fresh Groq quota) →
  screenshots → embed in README (`<!-- SCREENSHOTS -->` placeholder) → the LinkedIn post.
