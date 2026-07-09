"""Dataset generator module for creating Q&A pairs.

V2: Now supports both template-based and LLM-backed generation.
"""

import hashlib
import random
import time
from typing import Any, Callable

from distillery.schemas import (
    Dataset,
    DatasetIntent,
    DatasetOutput,
    Difficulty,
    GenerationRequest,
    QAPair,
    TrainingValue,
)


class TemplateGenerator:
    """Template-based generator for fallback and testing.
    
    This is the original V1 generator, now renamed and used as fallback
    when LLM is not available.
    """
    
    # Question templates by dataset type
    QUESTION_TEMPLATES: dict[str, list[str]] = {
        "bugfixing": [
            "How would you fix the bug where {scenario}?",
            "What is wrong with this code that causes {issue}?",
            "Debug the following code that produces {error}:",
            "Identify and fix the issue in this {language} function:",
            "The following code has a {bug_type} bug. How would you fix it?",
            "Why does this code throw {exception} and how to resolve it?",
            "Refactor this buggy code to handle {edge_case} correctly:",
            "What changes are needed to fix the {problem} in this snippet?",
        ],
        "testcase_generation": [
            "Write unit tests for a function that {functionality}.",
            "Generate test cases to verify {behavior}.",
            "What edge cases should be tested for {feature}?",
            "Create pytest tests for the following {component}:",
            "Write integration tests to validate {flow}.",
            "How would you test error handling for {scenario}?",
            "Design test cases that cover both positive and negative paths for {feature}.",
            "What mocking strategy would you use to test {component}?",
        ],
        "doc_generation": [
            "Write documentation for this {component_type}:",
            "Create a docstring for a function that {functionality}.",
            "Document the API endpoint that {action}.",
            "Write a README section explaining {feature}.",
            "Generate inline comments for this complex {algorithm}:",
            "Create usage examples for the {module} module.",
            "Write a changelog entry for {change}.",
            "Document the configuration options for {component}.",
        ],
        "code_review": [
            "Review this code for {aspect} issues:",
            "What improvements would you suggest for this {language} code?",
            "Identify potential security vulnerabilities in this snippet:",
            "How would you improve the readability of this function?",
            "What design pattern would better suit this implementation?",
            "Review this pull request for {concern}:",
            "Suggest performance optimizations for this code:",
            "What are the code smells in this implementation?",
        ],
        "refactoring": [
            "Refactor this code to improve {quality}:",
            "How would you restructure this to follow {principle}?",
            "Convert this procedural code to use {pattern}:",
            "Simplify this complex function while maintaining {behavior}:",
            "Extract common functionality from these {components}:",
            "Modernize this legacy code to use {feature}:",
            "Apply the {principle} principle to this code:",
            "Break down this monolithic function into smaller units:",
        ],
        "default": [
            "How would you implement {feature}?",
            "Explain the approach for {task}.",
            "What is the best practice for {scenario}?",
            "Write code to accomplish {goal}.",
            "Describe how to handle {situation}.",
            "What considerations are important for {topic}?",
            "Implement a solution for {problem}.",
            "How does {concept} work in this context?",
        ],
    }
    
    PLACEHOLDERS = {
        "scenario": ["user input validation fails", "data not persisting correctly", "API returns null", "async operation times out", "memory leak occurs"],
        "issue": ["incorrect output", "infinite loop", "null pointer exception", "race condition", "data corruption"],
        "error": ["TypeError", "ValueError", "KeyError", "IndexError", "AttributeError"],
        "language": ["Python", "JavaScript", "TypeScript", "Java", "Go"],
        "bug_type": ["logic", "off-by-one", "null reference", "type mismatch", "boundary"],
        "exception": ["IndexError", "KeyError", "RuntimeError", "ValueError", "TypeError"],
        "edge_case": ["empty input", "large datasets", "concurrent access", "malformed data", "timeout scenarios"],
        "problem": ["performance bottleneck", "memory leak", "race condition", "deadlock", "data inconsistency"],
        "functionality": ["calculates tax", "validates email", "parses JSON", "sorts data", "handles authentication"],
        "behavior": ["input validation", "error handling", "data transformation", "caching logic", "rate limiting"],
        "feature": ["user registration", "file upload", "search functionality", "notification system", "payment processing"],
        "component": ["API handler", "database service", "cache manager", "queue processor", "authentication module"],
        "flow": ["checkout process", "login workflow", "data sync", "report generation", "import pipeline"],
        "component_type": ["class", "function", "module", "API endpoint", "utility"],
        "action": ["creates a new user", "retrieves order details", "updates settings", "deletes records", "processes payments"],
        "algorithm": ["sorting implementation", "search algorithm", "graph traversal", "dynamic programming solution", "caching strategy"],
        "module": ["authentication", "database", "caching", "logging", "validation"],
        "change": ["adding retry logic", "fixing null handling", "improving performance", "adding new endpoint", "refactoring service"],
        "aspect": ["performance", "security", "maintainability", "readability", "testability"],
        "concern": ["breaking changes", "test coverage", "documentation", "error handling", "coding standards"],
        "quality": ["readability", "performance", "maintainability", "testability", "modularity"],
        "principle": ["DRY", "SOLID", "separation of concerns", "single responsibility", "dependency injection"],
        "pattern": ["factory pattern", "strategy pattern", "observer pattern", "decorator pattern", "repository pattern"],
        "components": ["service classes", "utility functions", "data models", "handlers", "validators"],
        "goal": ["data validation", "error recovery", "performance optimization", "security hardening", "code organization"],
        "task": ["implementing caching", "handling errors", "managing state", "processing data", "integrating APIs"],
        "situation": ["high traffic", "data migration", "system failure", "concurrent updates", "resource constraints"],
        "topic": ["scalability", "security", "performance", "maintainability", "reliability"],
        "concept": ["dependency injection", "memoization", "lazy loading", "event sourcing", "CQRS"],
    }
    
    CODE_SNIPPETS = [
        "def process_data(items):\n    return [x * 2 for x in items if x > 0]",
        "async function fetchUser(id) {\n  const response = await api.get(`/users/${id}`);\n  return response.data;\n}",
        "class DataProcessor:\n    def __init__(self, config):\n        self.config = config\n    \n    def run(self):\n        pass",
        "const validateInput = (input) => {\n  if (!input) throw new Error('Invalid input');\n  return input.trim();\n};",
        "def calculate_total(items, discount=0):\n    subtotal = sum(item.price for item in items)\n    return subtotal * (1 - discount)",
    ]
    
    def __init__(self, seed: int | None = None):
        self._random = random.Random(seed)
    
    def _fill_template(self, template: str) -> str:
        result = template
        for key, values in self.PLACEHOLDERS.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, self._random.choice(values), 1)
        return result
    
    def _generate_answer(self, question: str, dataset_type: str, constraints: dict) -> str:
        prefix = self._random.choice([
            "Here's how to approach this:\n\n",
            "The solution involves the following steps:\n\n",
            "To address this:\n\n",
            "Let me explain the approach:\n\n",
        ])
        
        difficulty = constraints.get("difficulty", "medium")
        code = self._random.choice(self.CODE_SNIPPETS)
        
        answer_parts = [
            "1. **Identify the root cause**: Analyze the problem systematically.",
            "2. **Validate inputs**: Ensure all inputs are properly validated.",
            "3. **Implement solution**: Apply the appropriate fix or implementation.",
            "4. **Add error handling**: Wrap operations in proper try-catch blocks.",
            "5. **Write tests**: Add tests to verify the solution works correctly.",
            "6. **Document changes**: Update documentation to reflect the changes.",
        ]
        
        steps = {"easy": 3, "medium": 4, "hard": 6}.get(difficulty, 4)
        return prefix + "\n".join(answer_parts[:steps]) + f"\n\n```python\n{code}\n```"
    
    def _generate_metadata(
        self, 
        question: str, 
        answer: str, 
        dataset_type: str, 
        index: int,
        intent_label: str = "",
    ) -> dict[str, Any]:
        content_hash = hashlib.md5(f"{question}{answer}".encode()).hexdigest()[:8]
        difficulty = self._random.choice(list(Difficulty))
        training_value = self._random.choice(list(TrainingValue))
        
        return {
            "id": f"{dataset_type}_{index:04d}_{content_hash}",
            "difficulty": difficulty.value,
            "intent_label": intent_label or f"{dataset_type}_intent_{index % 3}",
            "estimated_training_value": training_value.value,
            "source": "template",
            "has_code": "```" in answer,
        }
    
    def generate_intents(self, dataset_type: str, count: int = 4) -> list[DatasetIntent]:
        """Generate intents for a dataset type."""
        intent_templates = {
            "bugfixing": [
                DatasetIntent(label="error_diagnosis", description="Identify root cause of errors"),
                DatasetIntent(label="fix_implementation", description="Implement correct solution"),
                DatasetIntent(label="prevention", description="Best practices to prevent bugs"),
                DatasetIntent(label="edge_cases", description="Handle edge cases"),
            ],
            "testcase_generation": [
                DatasetIntent(label="unit_testing", description="Write isolated unit tests"),
                DatasetIntent(label="integration_testing", description="Test component interactions"),
                DatasetIntent(label="edge_case_testing", description="Cover boundary conditions"),
                DatasetIntent(label="mocking", description="Mock external dependencies"),
            ],
            "doc_generation": [
                DatasetIntent(label="api_docs", description="Document API endpoints"),
                DatasetIntent(label="code_docs", description="Write docstrings and comments"),
                DatasetIntent(label="guides", description="Create user guides"),
                DatasetIntent(label="examples", description="Provide usage examples"),
            ],
        }
        
        intents = intent_templates.get(dataset_type, [
            DatasetIntent(label="general", description="General purpose"),
        ])
        return intents[:count]
    
    def generate(self, request: GenerationRequest) -> DatasetOutput:
        datasets = []
        
        for dataset_type in request.dataset_types:
            items = []
            intents = self.generate_intents(dataset_type)
            intent_labels = [i.label for i in intents]
            templates = self.QUESTION_TEMPLATES.get(
                dataset_type, 
                self.QUESTION_TEMPLATES["default"]
            )
            
            for i in range(request.qa_per_type):
                template = templates[i % len(templates)]
                question = self._fill_template(template)
                constraints_dict = request.constraints.model_dump()
                answer = self._generate_answer(question, dataset_type, constraints_dict)
                intent_label = intent_labels[i % len(intent_labels)]
                metadata = self._generate_metadata(question, answer, dataset_type, i, intent_label)
                
                items.append(QAPair(question=question, answer=answer, metadata=metadata))
            
            datasets.append(Dataset(
                type=dataset_type, 
                items=items,
                intents=intent_labels,
            ))
        
        total_items = sum(len(d.items) for d in datasets)
        summary = (
            f"Generated {total_items} Q&A pairs across {len(datasets)} dataset types "
            f"using template-based generation. "
            f"Types: {', '.join(request.dataset_types)}."
        )
        
        return DatasetOutput(
            project_summary=summary,
            datasets=datasets,
            generation_method="template",
            llm_provider="",
        )


class LLMDatasetGenerator:
    """LLM-backed dataset generator for V2.
    
    Uses an LLM to:
    1. Propose dataset intents per type
    2. Generate diverse Q&A pairs per intent
    3. Include rich metadata for each item
    
    Falls back to template generation if LLM is unavailable.
    """
    
    def __init__(
        self, 
        llm_client=None,
        seed: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """Initialize the LLM generator.
        
        Args:
            llm_client: LLM client instance (uses get_llm_client() if None)
            seed: Random seed for reproducibility
            progress_callback: Optional callback for progress updates
        """
        self._seed = seed
        self._random = random.Random(seed)
        self._progress_callback = progress_callback or (lambda x: None)
        
        # Lazy load LLM client
        self._llm = llm_client
        
        # Keep template generator as fallback
        self._template_generator = TemplateGenerator(seed=seed)
    
    @property
    def llm(self):
        """Lazy-load LLM client."""
        if self._llm is None:
            from distillery.llm import get_llm_client
            self._llm = get_llm_client()
        return self._llm
    
    def _report_progress(self, message: str):
        """Report progress to callback."""
        self._progress_callback(message)
    
    def generate_intents(self, dataset_type: str, prompt: str) -> list[DatasetIntent]:
        """Generate intents for a dataset type using LLM."""
        self._report_progress(f"Generating intents for {dataset_type}...")
        
        # Use specialized intent generation for testcase_generation
        if dataset_type in ("testcase_generation", "testcase"):
            llm_prompt = f"""You are designing a fine-tuning dataset for a code LLM focused on generating pytest tests.

User Goal: {prompt}

Generate 6-8 distinct intents/themes for pytest test generation. Each intent should represent a different testing pattern or skill.

REQUIRED intents to include (pick at least 5):
- boundary_conditions: Testing edge cases and boundary values
- invalid_inputs: Handling invalid or malformed inputs
- parametrization: Data-driven tests with pytest.mark.parametrize
- exception_paths: Testing error handling with pytest.raises
- stateful_behavior: Testing state changes and side effects
- mocking: Mocking external dependencies with fixtures
- async_tests: Testing async functions with pytest-asyncio
- regression_tests: Preventing reintroduction of fixed bugs

Return JSON with this structure:
{{
  "intents": [
    {{"label": "intent_name", "description": "What this intent covers"}}
  ]
}}

Be specific. Each intent guides generation of tests with different pytest features."""
        else:
            llm_prompt = f"""You are designing a fine-tuning dataset for a code LLM.

Dataset Type: {dataset_type}
User Goal: {prompt}

Generate 4-5 distinct intents/themes for this dataset. Each intent should represent a different aspect or skill the model should learn.

Return JSON with this structure:
{{
  "intents": [
    {{"label": "intent_name", "description": "What this intent covers"}}
  ]
}}

Be specific and actionable. The intents should guide diverse Q&A generation."""

        try:
            response = self.llm.generate_json(llm_prompt)
            intents_data = response.get("intents", [])
            return [
                DatasetIntent(
                    label=i.get("label", f"intent_{idx}"),
                    description=i.get("description", ""),
                )
                for idx, i in enumerate(intents_data)
            ]
        except Exception as e:
            self._report_progress(f"LLM intent generation failed, using templates: {e}")
            return self._template_generator.generate_intents(dataset_type)
    
    def generate_qa_batch(
        self,
        dataset_type: str,
        intent: DatasetIntent,
        count: int,
        constraints: dict,
        existing_questions: list[str],
    ) -> list[QAPair]:
        """Generate a batch of Q&A pairs for a specific intent.
        
        Args:
            dataset_type: Type of dataset (e.g., "bugfixing")
            intent: The intent to generate for
            count: Number of pairs to generate
            constraints: User constraints
            existing_questions: Questions already generated (for diversity)
            
        Returns:
            List of QAPair objects
        """
        self._report_progress(f"Generating {count} Q&A pairs for intent: {intent.label}...")
        
        # Build context about existing questions for diversity
        existing_context = ""
        if existing_questions:
            sample = existing_questions[-5:]  # Last 5 for context
            existing_context = f"\n\nAlready generated questions (avoid similar ones):\n" + \
                "\n".join(f"- {q}" for q in sample)
        
        difficulty = constraints.get("difficulty", "medium")
        tone = constraints.get("tone", "technical")
        
        # Use specialized prompt for testcase_generation
        if dataset_type in ("testcase_generation", "testcase"):
            llm_prompt = self._get_testcase_generation_prompt(
                count, intent, difficulty, tone, existing_context
            )
        else:
            llm_prompt = f"""Generate {count} high-quality Q&A pairs for fine-tuning a code LLM.

Dataset Type: {dataset_type}
Intent: {intent.label} - {intent.description}
Difficulty: {difficulty}
Tone: {tone}
{existing_context}

Requirements:
1. Questions should be diverse and realistic
2. Answers should be detailed with code examples where appropriate
3. Each pair should teach a distinct concept or skill
4. Vary question structures (how, what, why, when, debug, implement, etc.)

Return JSON:
{{
  "items": [
    {{
      "question": "The question text",
      "answer": "Detailed answer with explanation and code if relevant",
      "metadata": {{
        "difficulty": "easy|medium|hard",
        "intent_label": "{intent.label}",
        "estimated_training_value": "low|medium|high",
        "source": "synthetic"
      }}
    }}
  ]
}}

Generate exactly {count} items. Make them production-quality."""

        try:
            response = self._generate_json_with_retry(llm_prompt)
            items_data = response.get("items", [])
            
            qa_pairs = []
            for idx, item in enumerate(items_data[:count]):
                metadata = item.get("metadata", {})
                metadata["id"] = self._generate_id(
                    item.get("question", ""),
                    item.get("answer", ""),
                    dataset_type,
                    len(existing_questions) + idx,
                )
                metadata["intent_label"] = intent.label
                metadata["source"] = "synthetic"
                
                qa_pairs.append(QAPair(
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    metadata=metadata,
                ))
            
            return qa_pairs
            
        except Exception as e:
            self._report_progress(f"LLM batch generation failed: {e}")
            # Fall back to template generation for this batch
            return self._generate_template_batch(dataset_type, intent, count, constraints)
    
    def _generate_json_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> dict:
        """Call the LLM with retries + exponential backoff for transient errors.

        Retries on rate limits (HTTP 429) and 5xx server errors, honoring a
        Retry-After header when present. Re-raises the last error after the
        retries are exhausted so the caller can fall back to templates.
        """
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self.llm.generate_json(prompt)
            except Exception as e:  # provider-agnostic: don't import httpx here
                last_error = e
                response = getattr(e, "response", None)
                status = getattr(response, "status_code", None)
                is_rate_limit = status == 429 or "429" in str(e)
                is_retryable = is_rate_limit or (status is not None and status >= 500)

                if attempt >= max_retries or not is_retryable:
                    raise

                delay = base_delay * (2 ** attempt)
                # Honor Retry-After (seconds) if the provider sent one.
                headers = getattr(response, "headers", None)
                if headers is not None:
                    retry_after = headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass

                self._report_progress(
                    f"LLM call {'rate-limited (429)' if is_rate_limit else f'failed ({status})'}; "
                    f"retry {attempt + 1}/{max_retries} in {delay:.0f}s..."
                )
                time.sleep(delay)

        # Defensive: loop always returns or raises, but satisfy type checkers.
        raise last_error  # type: ignore[misc]

    def _generate_template_batch(
        self,
        dataset_type: str,
        intent: DatasetIntent,
        count: int,
        constraints: dict,
    ) -> list[QAPair]:
        """Fallback template-based batch generation."""
        # Reuse template generator logic
        from distillery.schemas import UserConstraints
        request = GenerationRequest(
            prompt="",
            dataset_types=[dataset_type],
            qa_per_type=count,
            constraints=UserConstraints(**{
                k: v for k, v in constraints.items() 
                if k in UserConstraints.model_fields
            }),
        )
        output = self._template_generator.generate(request)
        return output.datasets[0].items if output.datasets else []
    
    def _get_testcase_generation_prompt(
        self,
        count: int,
        intent: DatasetIntent,
        difficulty: str,
        tone: str,
        existing_context: str,
    ) -> str:
        """Get specialized prompt for generating pytest test Q&A pairs.
        
        This prompt enforces the pytest contract:
        - At least one function with `def test_`
        - At least two `assert` statements
        - At least one of: pytest.mark.parametrize, pytest.raises, or fixture
        - Code inside a Python code block
        """
        # Map intents to specific pytest features to use
        intent_feature_map = {
            "boundary_conditions": "pytest.mark.parametrize with boundary values",
            "invalid_inputs": "pytest.raises for exception handling",
            "parametrization": "pytest.mark.parametrize with multiple test cases",
            "exception_paths": "pytest.raises with exception types and messages",
            "stateful_behavior": "fixtures for setup/teardown and state management",
            "mocking": "@pytest.fixture with mocked dependencies",
            "async_tests": "@pytest.mark.asyncio with async fixtures",
            "regression_tests": "parametrized tests covering regression scenarios",
        }
        
        feature_guidance = intent_feature_map.get(intent.label, "pytest fixtures or parametrize")
        
        return f"""Generate {count} high-quality pytest test Q&A pairs for fine-tuning a code LLM.

Dataset Type: testcase_generation
Intent: {intent.label} - {intent.description}
Difficulty: {difficulty}
Tone: {tone}
Focus: Use {feature_guidance}
{existing_context}

STRICT REQUIREMENTS - Each answer MUST include:
1. A Python code block (```python)
2. At least ONE test function starting with `def test_`
3. At least TWO `assert` statements
4. At least ONE of these pytest features:
   - @pytest.mark.parametrize for data-driven tests
   - pytest.raises for exception testing
   - @pytest.fixture or fixture parameters (like request, tmp_path, mocker)

Example structure:
```python
import pytest

@pytest.fixture
def sample_fixture():
    return SomeClass()

class TestFeature:
    @pytest.mark.parametrize("input,expected", [(1, 2), (3, 6)])
    def test_calculation(self, sample_fixture, input, expected):
        result = sample_fixture.calculate(input)
        assert result == expected
        assert isinstance(result, int)
    
    def test_error_handling(self, sample_fixture):
        with pytest.raises(ValueError, match="Invalid input"):
            sample_fixture.calculate(-1)
        assert sample_fixture.error_count > 0
```

Return JSON:
{{
  "items": [
    {{
      "question": "Question asking for specific pytest tests",
      "answer": "Explanation followed by pytest code meeting ALL requirements above",
      "metadata": {{
        "difficulty": "easy|medium|hard",
        "intent_label": "{intent.label}",
        "estimated_training_value": "low|medium|high",
        "source": "synthetic"
      }}
    }}
  ]
}}

Generate exactly {count} items. EVERY answer MUST include valid pytest code meeting ALL requirements."""
    
    def _generate_id(
        self, 
        question: str, 
        answer: str, 
        dataset_type: str, 
        index: int
    ) -> str:
        """Generate a unique ID for a Q&A pair."""
        content_hash = hashlib.md5(f"{question}{answer}".encode()).hexdigest()[:8]
        return f"{dataset_type}_{index:04d}_{content_hash}"
    
    def generate(
        self,
        request: GenerationRequest,
    ) -> DatasetOutput:
        """Generate datasets based on the request.
        
        Uses LLM when use_llm=True, otherwise falls back to templates.
        
        Args:
            request: The generation request
            
        Returns:
            DatasetOutput containing all generated datasets
        """
        # Check if we should use LLM
        if not request.use_llm:
            return self._template_generator.generate(request)
        
        datasets = []
        batch_size = request.batch_size or 10
        
        for dataset_type in request.dataset_types:
            self._report_progress(f"Processing dataset type: {dataset_type}")
            
            # Step 1: Generate intents
            intents = self.generate_intents(dataset_type, request.prompt)
            if not intents:
                # Fallback if no intents generated
                from distillery.schemas import DatasetIntent
                intents = [DatasetIntent(label="general", description="General purpose")]
            intent_labels = [i.label for i in intents]
            
            # Step 2: Generate Q&A pairs in batches, distributed across intents
            items: list[QAPair] = []
            existing_questions: list[str] = []
            
            # Distribute items across intents
            items_per_intent = request.qa_per_type // len(intents)
            remainder = request.qa_per_type % len(intents)
            
            for i, intent in enumerate(intents):
                # Add one extra item to first 'remainder' intents
                intent_count = items_per_intent + (1 if i < remainder else 0)
                
                # Generate in batches
                intent_items = []
                remaining = intent_count
                
                while remaining > 0:
                    batch_count = min(batch_size, remaining)
                    batch = self.generate_qa_batch(
                        dataset_type=dataset_type,
                        intent=intent,
                        count=batch_count,
                        constraints=request.constraints.model_dump(),
                        existing_questions=existing_questions,
                    )
                    
                    intent_items.extend(batch)
                    existing_questions.extend(item.question for item in batch)
                    remaining -= len(batch)
                
                items.extend(intent_items[:intent_count])
            
            datasets.append(Dataset(
                type=dataset_type,
                items=items,
                intents=intent_labels,
            ))
        
        total_items = sum(len(d.items) for d in datasets)
        summary = (
            f"Generated {total_items} Q&A pairs across {len(datasets)} dataset types "
            f"using LLM-backed generation (provider: {self.llm.provider_name}). "
            f"Types: {', '.join(request.dataset_types)}. "
            f"Model family: {request.constraints.model_family.value}."
        )
        
        return DatasetOutput(
            project_summary=summary,
            datasets=datasets,
            generation_method="llm",
            llm_provider=self.llm.provider_name,
        )


# =============================================================================
# Factory function for backward compatibility
# =============================================================================

class DatasetGenerator(LLMDatasetGenerator):
    """Main dataset generator class.
    
    This is an alias for LLMDatasetGenerator that maintains backward
    compatibility with V1 code.
    """
    pass
