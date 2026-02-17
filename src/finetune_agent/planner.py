"""Planner module for generating action plans.

V2: Now generates professional ML engineer-style documentation
with rationale, risks, and downstream use cases.
"""

from datetime import datetime

from finetune_agent.schemas import GenerationRequest, ModelFamily


class Planner:
    """Generates action plan documents based on user requirements.
    
    V2: Creates comprehensive documentation that reads like a real
    ML engineering design document.
    """
    
    # Model family descriptions
    MODEL_FAMILY_INFO = {
        ModelFamily.CODE_LLM: {
            "name": "Code Language Model",
            "description": "Optimized for code generation, completion, and understanding",
            "recommended_dataset_types": ["bugfixing", "code_review", "refactoring"],
            "training_notes": "Ensure code examples are syntactically correct and well-formatted",
        },
        ModelFamily.CHAT_LLM: {
            "name": "Conversational Language Model",
            "description": "Optimized for multi-turn dialogue and instruction following",
            "recommended_dataset_types": ["doc_generation", "general"],
            "training_notes": "Include diverse conversation styles and tones",
        },
        ModelFamily.CLASSIFIER: {
            "name": "Classification Model",
            "description": "Optimized for categorization and labeling tasks",
            "recommended_dataset_types": ["code_review"],
            "training_notes": "Ensure clear class boundaries and balanced examples",
        },
        ModelFamily.INSTRUCT: {
            "name": "Instruction-Following Model",
            "description": "Optimized for following complex instructions",
            "recommended_dataset_types": ["all"],
            "training_notes": "Vary instruction complexity and structure",
        },
        ModelFamily.OTHER: {
            "name": "Custom Model",
            "description": "Custom fine-tuning target",
            "recommended_dataset_types": ["all"],
            "training_notes": "Adapt data format to model requirements",
        },
    }
    
    def __init__(self, llm_client=None):
        """Initialize the planner."""
        self._llm = llm_client
    
    def _infer_downstream_use(self, request: GenerationRequest) -> str:
        """Infer the downstream use case from the request."""
        prompt_lower = request.prompt.lower()
        dataset_types = [t.lower() for t in request.dataset_types]
        
        if "debug" in prompt_lower or "bugfix" in prompt_lower or "bugfixing" in dataset_types:
            return "Code debugging and error resolution assistance"
        elif "test" in prompt_lower or "testcase" in dataset_types:
            return "Automated test generation and quality assurance"
        elif "doc" in prompt_lower or "doc_generation" in dataset_types:
            return "Documentation and code explanation generation"
        elif "review" in prompt_lower or "code_review" in dataset_types:
            return "Code review and quality assessment"
        elif "refactor" in prompt_lower or "refactoring" in dataset_types:
            return "Code refactoring and modernization"
        else:
            return "General code assistance and instruction following"
    
    def _get_dataset_contract_section(self, request: GenerationRequest) -> str:
        """Generate Dataset Contract section for specific dataset types.
        
        This section specifies strict requirements for answer format.
        """
        sections = []
        
        # testcase_generation contract
        if any(dt in ("testcase_generation", "testcase") for dt in request.dataset_types):
            sections.append("""
### 2.4 Dataset Contract: testcase_generation

**STRICT REQUIREMENTS** — Every answer MUST include:

1. **Python Code Block**: Answer must contain a ` ```python ... ``` ` code block
2. **Test Function**: At least one function starting with `def test_`
3. **Assertions**: At least 2 `assert` statements within the code block
4. **Pytest Feature**: At least one of:
   - `@pytest.mark.parametrize` for data-driven tests
   - `pytest.raises` for exception testing
   - Fixture usage (e.g., `tmp_path`, `mocker`, `capsys`, or `@pytest.fixture`)

**Intents for testcase_generation** (6-8 required):
- `boundary_conditions` — Testing edge cases and boundary values
- `invalid_inputs` — Handling invalid or malformed inputs
- `parametrization` — Data-driven tests with pytest.mark.parametrize
- `exception_paths` — Testing error handling with pytest.raises
- `stateful_behavior` — Testing state changes and side effects
- `mocking` — Mocking external dependencies with fixtures
- `async_tests` — Testing async functions with pytest-asyncio
- `regression_tests` — Preventing reintroduction of fixed bugs

**Example Schema for testcase_generation**:
```json
{
  "question": "Write pytest tests for a user authentication function that validates email/password and handles invalid inputs",
  "answer": "Here's a comprehensive test suite:\\n\\n```python\\nimport pytest\\n\\nclass TestUserAuth:\\n    @pytest.fixture\\n    def auth_service(self):\\n        return AuthService()\\n\\n    @pytest.mark.parametrize(\\"email,password,expected\\", [\\n        (\\"valid@example.com\\", \\"Password123!\\", True),\\n        (\\"invalid\\", \\"pass\\", False),\\n    ])\\n    def test_login_validation(self, auth_service, email, password, expected):\\n        result = auth_service.validate(email, password)\\n        assert result == expected\\n        assert auth_service.error_count >= 0\\n\\n    def test_invalid_email_raises(self, auth_service):\\n        with pytest.raises(ValueError, match=\\"Invalid email\\"):\\n            auth_service.validate(\\"not-an-email\\", \\"pass\\")\\n        assert auth_service.last_error is not None\\n```",
  "metadata": {
    "difficulty": "medium",
    "intent_label": "parametrization",
    "estimated_training_value": "high",
    "source": "synthetic"
  }
}
```
""")
        
        return "\n".join(sections)
    
    def _identify_risks(self, request: GenerationRequest) -> list[dict]:
        """Identify potential risks with the dataset generation."""
        risks = []
        
        # Data quality risks
        if request.qa_per_type < 10:
            risks.append({
                "risk": "Insufficient training data",
                "severity": "High",
                "mitigation": "Increase qa_per_type to at least 20-50 per type",
            })
        
        if request.qa_per_type > 100:
            risks.append({
                "risk": "Potential quality degradation at scale",
                "severity": "Medium",
                "mitigation": "Apply aggressive filtering and manual review",
            })
        
        # Diversity risks
        if len(request.dataset_types) < 2:
            risks.append({
                "risk": "Limited task diversity",
                "severity": "Medium",
                "mitigation": "Consider adding complementary dataset types",
            })
        
        # Synthetic data risks
        risks.append({
            "risk": "Synthetic data may lack real-world edge cases",
            "severity": "Medium",
            "mitigation": "Supplement with real examples from production codebases",
        })
        
        # Hallucination risk
        risks.append({
            "risk": "LLM-generated answers may contain errors",
            "severity": "Medium",
            "mitigation": "Human review of generated content before training",
        })
        
        # Data leakage
        risks.append({
            "risk": "Potential training/eval data leakage",
            "severity": "Low",
            "mitigation": "Ensure generated content is distinct from evaluation sets",
        })
        
        return risks
    
    def generate_action_plan(self, request: GenerationRequest) -> str:
        """Generate an action plan markdown document.
        
        V2: Creates a professional ML engineering document with:
        - Target model analysis
        - Dataset design rationale
        - Risk assessment
        - Implementation roadmap
        
        Args:
            request: The generation request with user requirements
            
        Returns:
            Markdown string containing the action plan
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_info = self.MODEL_FAMILY_INFO.get(
            request.constraints.model_family,
            self.MODEL_FAMILY_INFO[ModelFamily.OTHER]
        )
        
        # Format dataset types
        dataset_types_list = "\n".join(f"- **{dt}**" for dt in request.dataset_types)
        
        # Build constraints section
        constraints_lines = []
        constraints_lines.append(f"- **Tone**: {request.constraints.tone}")
        constraints_lines.append(f"- **Difficulty**: {request.constraints.difficulty}")
        constraints_lines.append(f"- **Model Family**: {model_info['name']}")
        if request.constraints.domain:
            constraints_lines.append(f"- **Domain**: {request.constraints.domain}")
        if request.constraints.aggressive_filtering:
            constraints_lines.append("- **Filtering**: Aggressive (stricter quality thresholds)")
        if request.constraints.additional_notes:
            constraints_lines.append(f"- **Additional Notes**: {request.constraints.additional_notes}")
        
        constraints_section = "\n".join(constraints_lines)
        
        # Identify downstream use
        downstream_use = self._infer_downstream_use(request)
        
        # Identify risks
        risks = self._identify_risks(request)
        risk_rows = "\n".join(
            f"| {r['risk']} | {r['severity']} | {r['mitigation']} |"
            for r in risks
        )
        
        # Calculate expected metrics
        total_items = len(request.dataset_types) * request.qa_per_type
        estimated_tokens = total_items * 500  # Rough estimate
        
        action_plan = f"""# Fine-Tuning Dataset Engineering Plan

**Document Version**: 2.0  
**Generated**: {timestamp}  
**Status**: Ready for Generation

---

## Executive Summary

This document outlines the engineering plan for generating a fine-tuning dataset
based on the following objective:

> {request.prompt}

**Target Model**: {model_info['name']}  
**Expected Output**: {total_items} Q&A pairs across {len(request.dataset_types)} dataset types  
**Downstream Use**: {downstream_use}

---

## 1. Target Model Analysis

### 1.1 Model Family

**{model_info['name']}**

{model_info['description']}

### 1.2 Training Considerations

{model_info['training_notes']}

### 1.3 Expected Use Cases

Based on the configuration, this dataset is optimized for:

- **Primary**: {downstream_use}
- **Dataset Types**: {', '.join(request.dataset_types)}
- **Difficulty Level**: {request.constraints.difficulty.capitalize()}

---

## 2. Dataset Design

### 2.1 Overview

| Metric | Value |
|--------|-------|
| Dataset Types | {len(request.dataset_types)} |
| Items per Type | {request.qa_per_type} |
| Total Items | {total_items} |
| Est. Token Count | ~{estimated_tokens:,} |
| Generation Method | LLM-backed with template fallback |

### 2.2 Dataset Types

{dataset_types_list}

### 2.3 Constraints & Requirements

{constraints_section}

{self._get_dataset_contract_section(request)}

---

## 3. Dataset Design Rationale

### 3.1 Why These Dataset Types?

The selected dataset types align with the stated objective:

1. **Skill Coverage**: Each type teaches a distinct skill (debugging, testing, documentation, etc.)
2. **Progressive Difficulty**: Items range from basic to advanced within each type
3. **Practical Relevance**: Types reflect real-world development workflows

### 3.2 Intent-Based Generation

For each dataset type, we generate 4-5 distinct intents to ensure variety:

- Each intent represents a different aspect of the skill
- Q&A pairs are distributed across intents
- This prevents mode collapse during fine-tuning

### 3.3 Quality Assurance

The generation pipeline includes:

1. **Intent Planning**: LLM proposes diverse intents per dataset type
2. **Batch Generation**: Items generated in batches of {request.batch_size} to maintain quality
3. **Self-Critique**: Automated review identifies duplicates and low-quality items
4. **Filtering**: Rejected items are optionally regenerated

---

## 4. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
{risk_rows}

---

## 5. Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GENERATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌────────────┐    ┌─────────┐    ┌────────────┐   │
│  │ Planner  │ -> │ Generator  │ -> │ Critic  │ -> │ Evaluator  │   │
│  └──────────┘    └────────────┘    └─────────┘    └────────────┘   │
│       │                │                │               │           │
│       v                v                v               v           │
│  action_plan.md   dataset.json    (filtering)    evaluation.json   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.1 Phase 1: Planning (Complete)
- ✅ Analyze user requirements
- ✅ Determine target model characteristics
- ✅ Identify risks and mitigations
- ✅ Generate this action plan

### 5.2 Phase 2: Generation (In Progress)
- ⏳ Generate intents for each dataset type
- ⏳ Create Q&A pairs per intent
- ⏳ Attach metadata (difficulty, training value, source)

### 5.3 Phase 3: Quality Control
- ⏳ Run self-critique agent
- ⏳ Filter near-duplicates and low-quality items
- ⏳ Optionally regenerate rejected items

### 5.4 Phase 4: Evaluation
- ⏳ Calculate uniqueness scores (lexical + structural + conceptual)
- ⏳ Compute health metrics
- ⏳ Generate actionable feedback

---

## 6. Expected Outputs

| File | Description |
|------|-------------|
| `action_plan.md` | This document |
| `dataset.json` | All Q&A pairs with metadata |
| `evaluation.json` | Quality scores, metrics, and feedback |

### 6.1 Dataset JSON Schema

```json
{{
  "project_summary": "...",
  "datasets": [
    {{
      "type": "bugfixing",
      "intents": ["error_diagnosis", "fix_implementation", ...],
      "items": [
        {{
          "question": "...",
          "answer": "...",
          "metadata": {{
            "id": "bugfixing_0001_abc123",
            "difficulty": "medium",
            "intent_label": "error_diagnosis",
            "estimated_training_value": "high",
            "source": "synthetic"
          }}
        }}
      ]
    }}
  ],
  "generation_method": "llm",
  "llm_provider": "mock"
}}
```

---

## 7. Recommendations

### 7.1 Before Training

1. **Manual Review**: Sample 10-20% of generated items for accuracy
2. **Domain Expert Check**: Have a subject matter expert validate technical content
3. **Deduplication**: Run additional deduplication against existing training data

### 7.2 During Training

1. **Validation Split**: Reserve 10-15% for validation
2. **Early Stopping**: Monitor validation loss to prevent overfitting
3. **Learning Rate**: Start with a low learning rate for synthetic data

### 7.3 After Training

1. **Evaluation**: Test on held-out examples not from this generation run
2. **Human Eval**: Conduct human preference evaluation on key use cases
3. **Iterate**: Use evaluation results to inform next dataset generation

---

## 8. Roadmap

### 8.1 Current Capabilities (V2)
- ✅ LLM-backed generation with template fallback
- ✅ Self-critique and filtering
- ✅ Multi-metric uniqueness scoring
- ✅ Health metrics and warnings

### 8.2 Planned Improvements
- [ ] GitHub repository ingestion for context-aware generation
- [ ] Embedding-based semantic deduplication
- [ ] Active learning for targeted data generation
- [ ] Multi-turn conversation dataset support
- [ ] Direct export to HuggingFace format

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **Intent** | A specific skill or concept covered by Q&A pairs |
| **Training Value** | Estimated usefulness for model learning (low/medium/high) |
| **Uniqueness Score** | Metric combining lexical, structural, and conceptual diversity |
| **Health Metrics** | Dataset statistics like difficulty distribution and code coverage |

---

*Generated by Finetune Agent v2.0*  
*This tool does NOT train models. It accelerates finetuning engineering.*
"""
        return action_plan
