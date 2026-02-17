# Finetune Agent v2.0

**An Agentic AI Assistant for Finetuning Engineering**

> **This tool does NOT train models. It accelerates finetuning engineering.**

---

## Why This Tool Exists

Fine-tuning LLMs is powerful but creating high-quality training datasets is painful:

- **Manual curation is slow** - Writing thousands of Q&A pairs takes weeks
- **Quality is inconsistent** - Hard to maintain diversity and avoid repetition
- **Evaluation is subjective** - No clear metrics for dataset quality
- **Iteration is expensive** - Each attempt requires significant effort

Finetune Agent solves these problems with an **agentic approach**:

1. **Automated Generation** - LLM-backed Q&A pair creation with intent-based diversity
2. **Self-Critique** - Built-in quality review that identifies duplicates and weak examples
3. **Quantitative Scoring** - Multi-metric evaluation with lexical, structural, and conceptual analysis
4. **Fast Iteration** - Generate, evaluate, and refine in minutes, not days

---

## How Agentic Generation Works

Finetune Agent uses a **Planner → Generator → Critic → Evaluator** pipeline:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         AGENTIC PIPELINE                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────┐    ┌────────────┐    ┌─────────┐    ┌────────────┐    │
│  │ Planner  │ -> │ Generator  │ -> │ Critic  │ -> │ Evaluator  │    │
│  └──────────┘    └────────────┘    └─────────┘    └────────────┘    │
│       │                │                │               │            │
│       v                v                v               v            │
│  action_plan.md   dataset.json    (filtering)    evaluation.json    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 1. Planner

Creates a professional engineering document with:
- Target model analysis
- Dataset design rationale
- Risk assessment
- Implementation roadmap

### 2. Generator

Uses LLM (or template fallback) to:
- Propose diverse intents per dataset type
- Generate Q&A pairs with rich metadata
- Batch generation to maintain quality

### 3. Critic

Self-review agent that:
- Identifies near-duplicate questions
- Flags trivial or low-signal answers
- Detects quality issues

### 4. Evaluator

Scores datasets using:
- Lexical analysis (TF-IDF, n-grams)
- Structural variety (question types, length distribution)
- Conceptual diversity (LLM-assisted semantic analysis)

---

## How Quality Scoring Works

### Uniqueness Score (0-100)

Combines three metrics with weighted average:

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| **Lexical** | 50% | Vocabulary diversity, TF-IDF similarity, n-gram overlap |
| **Structural** | 30% | Question word variety, length distribution |
| **Conceptual** | 20% | LLM-assessed semantic diversity |

```
uniqueness_score = 0.5 * lexical + 0.3 * structural + 0.2 * conceptual
```

### Overall Rating (0-100)

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| **Uniqueness** | 40% | Average uniqueness across datasets |
| **Length Sanity** | 25% | Appropriate question/answer lengths |
| **Coverage** | 35% | Even distribution across types |

### Health Metrics

- Average answer length
- Difficulty distribution (easy/medium/hard)
- Intent coverage score
- Code presence percentage

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd finetune-agent

# Install with pip
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

---

## How to Run

### Interactive CLI

```bash
python -m finetune_agent
```

### With LLM Provider (Recommended)

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-api-key
python -m finetune_agent
```

### Mock Mode (No API Key Needed)

```bash
# Runs with deterministic mock LLM (default)
python -m finetune_agent
```

### Streamlit Web UI

```bash
# Run the web interface
streamlit run src/finetune_agent/ui/app.py
```

The Streamlit UI provides:
- Visual configuration for all generation parameters
- Tabbed output view (Action Plan, Dataset, Critique, Evaluation)
- Download buttons for generated artifacts
- Run history from memory store

---

## Example Run

```
+---------------------------------------------------------------+
|                  FINETUNE AGENT v2.0                          |
|         Agentic AI for Finetuning Engineering                 |
+---------------------------------------------------------------+

Step 1: Describe your fine-tuning goal
> A Python debugging assistant that helps fix common errors

Step 2: Select target model family
> 1. Code LLM

Step 3: Choose dataset types
> bugfixing,testcase_generation

Step 4: Set dataset size
> 15

Step 5: Set generation constraints
> technical, medium difficulty

Step 6: Quality filtering
> Enable aggressive filtering? No

Starting Generation Pipeline...
Phase 1: Generating action plan...
Phase 2: Generating datasets...
Phase 3: Running self-critique...
Phase 5: Filtering rejected items...
Phase 6: Evaluating dataset quality...

Generation Complete!

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Dataset Type       ┃ Items ┃ Uniqueness ┃ Lexical ┃ Structural ┃ Conceptual ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ bugfixing          │    15 │      78.5  │   82.3  │      71.2  │      75.0  │
│ testcase_generation│    15 │      81.2  │   85.1  │      74.5  │      80.0  │
└────────────────────┴───────┴────────────┴─────────┴────────────┴────────────┘

Overall Rating: 76.4/100

Output saved to: artifacts/20260124_163045/
```

---

## Example Dataset JSON

```json
{
  "project_summary": "Generated 30 Q&A pairs across 2 dataset types...",
  "datasets": [
    {
      "type": "bugfixing",
      "intents": ["error_diagnosis", "fix_implementation", "prevention"],
      "items": [
        {
          "question": "How do I fix a NoneType error when accessing dictionary keys?",
          "answer": "Here's how to approach this:\n\n1. **Check for None before access**...",
          "metadata": {
            "id": "bugfixing_0001_a1b2c3d4",
            "difficulty": "medium",
            "intent_label": "error_diagnosis",
            "estimated_training_value": "high",
            "source": "synthetic"
          }
        }
      ]
    }
  ],
  "generation_method": "llm",
  "llm_provider": "mock"
}
```

---

## Output Files

| File | Description |
|------|-------------|
| `action_plan.md` | Professional engineering document with rationale and risks |
| `dataset.json` | All Q&A pairs with rich metadata |
| `evaluation.json` | Quality scores, health metrics, and feedback |
| `critique.json` | Self-critique results with reject indices |
| `debug.json` | **Troubleshooting info** - generation counts, rejection reasons |
| `dataset_qa.jsonl` | Simple Q&A format for training |
| `dataset_instruct.jsonl` | Alpaca-style instruction format |
| `dataset_chat.jsonl` | OpenAI chat format |
| `golden_set.jsonl` | Curated top items for evaluation |

---

## Export Formats

The agent automatically exports to multiple JSONL formats commonly used in fine-tuning workflows:

### Simple Q&A (`dataset_qa.jsonl`)

```json
{"question": "How do I fix a null pointer?", "answer": "Check for null before accessing...", "metadata": {"difficulty": "medium", ...}}
```

Best for: Simple supervised fine-tuning, RAG training.

### Instruction Format (`dataset_instruct.jsonl`)

```json
{"instruction": "How do I fix a null pointer?", "input": "", "output": "Check for null before accessing...", "metadata": {...}}
```

Best for: Alpaca-style instruction tuning, LLaMA fine-tuning.

### Chat Format (`dataset_chat.jsonl`)

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "How do I fix a null pointer?"}, {"role": "assistant", "content": "Check for null before accessing..."}], "metadata": {...}}
```

Best for: OpenAI chat fine-tuning, conversational models.

### Golden Set (`golden_set.jsonl`)

A curated subset of the best items for evaluation:
- **Top 10 most unique** - Highest diversity scores
- **Top 10 hardest** - Most challenging difficulty
- **Top 10 code-heavy** - Most code in answers

No duplicates across categories. Use for validation/evaluation sets.

---

## Advanced Constraints

Configure quality controls for production datasets:

| Constraint | Default | Description |
|------------|---------|-------------|
| `min_answer_length` | 50 | Minimum answer length in characters |
| `similarity_threshold` | 0.7 | Jaccard similarity for duplicate detection (0.0-1.0) |
| `require_code_ratio` | 0 | Minimum % of answers with code (for code LLMs) |
| `banned_phrases` | [] | Phrases that critic should flag |
| `difficulty_distribution` | 30/50/20 | Target % for easy/medium/hard |

Example in CLI:

```
Step 7: Advanced constraints (optional)
Configure advanced quality constraints? [y/N]: y

Minimum answer length (chars) [50]: 100
Similarity threshold for duplicates (0.0-1.0) [0.7]: 0.8
Minimum code ratio (0-100%) [0]: 50
Banned phrases (comma-separated, or blank): TODO, FIXME, placeholder
Set custom difficulty distribution? [y/N]: y
  Easy % [30]: 20
  Medium % [50]: 50
  Hard % [20]: 30
```

---

## Configuration

Create a `.env` file:

```env
# LLM Provider (openai or mock)
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key

# Optional: Custom OpenAI settings
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Redis for memory persistence (optional)
REDIS_URL=redis://localhost:6379/0
```

---

## Project Structure

```
finetune-agent/
├── src/finetune_agent/
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── cli.py                # Interactive CLI
│   ├── agent.py              # Pipeline orchestration
│   ├── planner.py            # Action plan generation
│   ├── dataset_generator.py  # Q&A generation (LLM + template)
│   ├── critic.py             # Self-critique agent
│   ├── evaluator.py          # Quality scoring
│   ├── schemas.py            # Pydantic models
│   ├── utils.py              # Utilities
│   ├── exporter.py           # JSONL export and golden set sampling
│   ├── llm/
│   │   ├── __init__.py       # LLM client factory
│   │   ├── base.py           # Abstract interface
│   │   ├── openai.py         # OpenAI client
│   │   └── mock.py           # Mock client for testing
│   ├── ui/
│   │   ├── __init__.py       # UI module
│   │   └── app.py            # Streamlit app
│   └── memory/
│       ├── __init__.py
│       ├── store.py          # Abstract interface
│       ├── redis_store.py    # Redis backend
│       └── local_store.py    # JSON file backend
├── tests/
│   ├── test_evaluator.py
│   ├── test_generator.py
│   └── test_critic.py
├── artifacts/                 # Generated outputs
├── pyproject.toml
└── README.md
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=finetune_agent

# Run specific test file
pytest tests/test_evaluator.py -v

# Run V2 critic tests
pytest tests/test_critic.py -v
```

---

## Troubleshooting

### Total Items = 0

If the agent generates 0 items, check `debug.json` in the artifacts folder:

```json
{
  "requested_count_per_type": 10,
  "generated_count_before_critique": {"bugfixing": 10, "testcase_generation": 10},
  "rejected_count": {"bugfixing": 0, "testcase_generation": 10},
  "top_rejection_reasons": {
    "testcase_generation": {
      "missing_code_block": 5,
      "missing_test_function": 3,
      "missing_pytest_feature": 2
    }
  },
  "refill_iterations_run": 3,
  "final_count": {"bugfixing": 10, "testcase_generation": 0}
}
```

#### Common Causes

1. **testcase_generation contract violations** - Answers must include proper pytest code:
   - `def test_` function inside a code block
   - At least 2 `assert` statements  
   - pytest feature (parametrize, raises, or fixture)

2. **Min answer length too high** - Relax `min_answer_length` constraint

3. **Similarity threshold too strict** - Increase `similarity_threshold` (e.g., 0.8)

4. **Aggressive filtering enabled** - Try disabling it

#### Quick Fixes

```bash
# Try with relaxed constraints
python -m finetune_agent
# When prompted, set:
# - min_answer_length: 0
# - similarity_threshold: 0.9
# - aggressive_filtering: No
```

### Mock Mode

If you don't have an API key, the agent uses mock mode by default. Mock mode:
- Generates deterministic, template-based outputs
- Is useful for testing the pipeline
- May produce lower-quality results than real LLM

---

## Roadmap

### Current (v2.0)
- ✅ LLM-backed generation with template fallback
- ✅ Self-critique agent for quality filtering
- ✅ Multi-metric uniqueness scoring (lexical + structural + conceptual)
- ✅ Health metrics and warnings
- ✅ Professional action plan documents

### Planned (v2.x)
- ✅ Streamlit Web UI for visual interaction
- [ ] GitHub repository ingestion for context-aware generation
- [ ] Embedding-based semantic deduplication
- [ ] Active learning for targeted data generation
- [ ] Multi-turn conversation dataset support

### Future (v3.0)
- [ ] Direct export to HuggingFace format
- [ ] Integration with fine-tuning platforms
- [ ] Automated difficulty calibration
- [ ] Real-time quality monitoring during training

---

## License

MIT License

---

Built with ❤️ for the fine-tuning community
