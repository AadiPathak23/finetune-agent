"""Mock LLM client for testing and development without API access."""

import hashlib
import json
import random
from typing import Any

from .base import LLMClient


class MockLLMClient(LLMClient):
    """Deterministic mock LLM client for testing.
    
    Generates plausible responses based on prompt content using
    templates and controlled randomization. Useful for:
    - Running tests without API access
    - Development without burning API credits
    - Reproducible test scenarios
    """
    
    def __init__(self, seed: int | None = None):
        """Initialize mock client with optional seed for reproducibility."""
        self._seed = seed
        self._random = random.Random(seed)
        self._call_count = 0
    
    @property
    def provider_name(self) -> str:
        return "mock"
    
    def _get_seeded_random(self, prompt: str) -> random.Random:
        """Get a random instance seeded by prompt content for consistency."""
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        combined_seed = (self._seed or 0) + prompt_hash + self._call_count
        self._call_count += 1
        return random.Random(combined_seed)
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate mock text response based on prompt content."""
        rng = self._get_seeded_random(prompt)
        
        prompt_lower = prompt.lower()
        
        # Detect prompt type and generate appropriate response
        if "uniqueness" in prompt_lower or "diversity" in prompt_lower:
            return self._generate_uniqueness_evaluation(rng, prompt)
        elif "critique" in prompt_lower or "review" in prompt_lower:
            return self._generate_critique(rng, prompt)
        elif "intent" in prompt_lower and "dataset" in prompt_lower:
            return self._generate_intents(rng, prompt)
        else:
            return self._generate_generic(rng, prompt)
    
    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        """Generate mock JSON response."""
        rng = self._get_seeded_random(prompt)
        prompt_lower = prompt.lower()
        
        # Detect what kind of JSON is expected
        if "q&a" in prompt_lower or "question" in prompt_lower and "answer" in prompt_lower:
            return self._generate_qa_json(rng, prompt)
        elif "intent" in prompt_lower:
            return self._generate_intents_json(rng, prompt)
        elif "critique" in prompt_lower or "review" in prompt_lower:
            return self._generate_critique_json(rng, prompt)
        elif "uniqueness" in prompt_lower or "conceptual" in prompt_lower:
            return self._generate_uniqueness_json(rng, prompt)
        else:
            return self._generate_generic_json(rng, prompt)
    
    def _generate_uniqueness_evaluation(self, rng: random.Random, prompt: str) -> str:
        """Generate uniqueness evaluation response."""
        score = rng.randint(55, 85)
        return f"""Based on my analysis of the provided dataset:

**Conceptual Diversity Score: {score}/100**

The dataset shows {'good' if score > 70 else 'moderate'} conceptual diversity. 
The questions cover different aspects of the target domain, though some 
thematic overlap exists. The answers demonstrate varied approaches and 
explanations.

Key observations:
- Question variety: {'Strong' if score > 75 else 'Adequate'}
- Answer depth: {'Comprehensive' if score > 70 else 'Sufficient'}  
- Topic coverage: {'Broad' if score > 65 else 'Focused'}
"""
    
    def _generate_critique(self, rng: random.Random, prompt: str) -> str:
        """Generate critique response."""
        issues = rng.randint(0, 3)
        return f"""Dataset Quality Review:

Found {issues} potential issues:
{'- Some questions may be too similar in structure' if issues > 0 else ''}
{'- A few answers could be more detailed' if issues > 1 else ''}
{'- Consider diversifying the difficulty levels' if issues > 2 else ''}

Overall: The dataset is {'acceptable' if issues < 2 else 'needs improvement'} for fine-tuning purposes.
"""
    
    def _generate_intents(self, rng: random.Random, prompt: str) -> str:
        """Generate dataset intents."""
        return """Proposed intents for this dataset type:
1. Error diagnosis and root cause analysis
2. Solution implementation with code examples
3. Best practices and prevention strategies
4. Edge case handling
5. Performance optimization related fixes"""
    
    def _generate_generic(self, rng: random.Random, prompt: str) -> str:
        """Generate generic response."""
        return "This is a mock LLM response for testing purposes. In production, this would contain meaningful generated content based on the prompt."
    
    def _generate_qa_json(self, rng: random.Random, prompt: str) -> dict[str, Any]:
        """Generate Q&A pairs JSON."""
        # Determine count from prompt
        count = 5
        for word in prompt.split():
            if word.isdigit():
                count = min(int(word), 20)
                break
        
        # Detect dataset type from prompt
        dataset_type = "general"
        prompt_lower = prompt.lower()
        # Check testcase_generation first (more specific match)
        if "testcase_generation" in prompt_lower or "testcase generation" in prompt_lower:
            dataset_type = "testcase_generation"
        else:
            for dt in ["bugfixing", "testcase", "doc", "review", "refactor"]:
                if dt in prompt_lower:
                    dataset_type = dt
                    break
        
        # Extract intent label from prompt if present
        intent_label = None
        for intent in ["boundary_conditions", "invalid_inputs", "parametrization", 
                       "exception_paths", "stateful_behavior", "mocking", 
                       "async_tests", "regression_tests"]:
            if intent in prompt_lower:
                intent_label = intent
                break
        
        items = []
        for i in range(count):
            difficulty = rng.choice(["easy", "medium", "hard"])
            training_value = rng.choice(["low", "medium", "high"])
            
            # For testcase_generation, use specialized generation
            if dataset_type in ("testcase", "testcase_generation"):
                question, answer = self._generate_pytest_qa(rng, i, intent_label, difficulty)
                final_intent = intent_label or f"testcase_intent_{i % 5}"
            else:
                question = self._generate_question(rng, dataset_type, i)
                answer = self._generate_answer(rng, dataset_type, difficulty)
                final_intent = intent_label or f"{dataset_type}_intent_{i % 3}"
            
            items.append({
                "question": question,
                "answer": answer,
                "metadata": {
                    "difficulty": difficulty,
                    "intent_label": final_intent,
                    "estimated_training_value": training_value,
                    "source": "synthetic",
                }
            })
        
        return {"items": items}
    
    def _generate_question(self, rng: random.Random, dtype: str, index: int) -> str:
        """Generate a realistic question based on dataset type."""
        questions = {
            "bugfixing": [
                "How do I fix a NoneType error when accessing dictionary keys?",
                "What causes 'index out of range' errors in list operations?",
                "How to resolve circular import issues in Python modules?",
                "Why does my async function return a coroutine instead of the result?",
                "How to fix memory leaks in long-running Python processes?",
                "What causes 'maximum recursion depth exceeded' and how to fix it?",
                "How to debug intermittent connection timeout errors?",
                "Why is my regex not matching expected patterns?",
                "How to fix Unicode encoding errors when reading files?",
                "What causes race conditions in multi-threaded code?",
            ],
            "testcase": [
                "Write unit tests for a user authentication function.",
                "How to test async functions with pytest?",
                "Create integration tests for a REST API endpoint.",
                "What test cases cover edge cases for date parsing?",
                "How to mock database connections in unit tests?",
                "Write property-based tests for a sorting algorithm.",
                "How to test error handling in file operations?",
                "Create tests for a caching decorator function.",
                "What tests verify thread-safety of a singleton class?",
                "How to test WebSocket connections?",
            ],
            "doc": [
                "Write documentation for a data processing pipeline.",
                "Create API documentation for a REST endpoint.",
                "Document the architecture of a microservices system.",
                "Write a README for an open-source CLI tool.",
                "Create docstrings for a machine learning model class.",
                "Document configuration options for a web application.",
                "Write changelog entries for a major version release.",
                "Create user guide for a database migration tool.",
                "Document error codes and their meanings.",
                "Write contributing guidelines for a project.",
            ],
            "review": [
                "Review this code for security vulnerabilities.",
                "What improvements would make this function more maintainable?",
                "Identify performance bottlenecks in this database query.",
                "Review error handling in this API client implementation.",
                "What design patterns would improve this class structure?",
                "Review this authentication flow for security issues.",
                "Identify code smells in this service layer.",
                "Review logging practices in this application.",
                "What refactoring would reduce complexity here?",
                "Review test coverage gaps in this module.",
            ],
            "refactor": [
                "Refactor this monolithic function into smaller units.",
                "Apply SOLID principles to this class hierarchy.",
                "Convert callback-based code to async/await.",
                "Extract common logic into reusable utilities.",
                "Refactor to use dependency injection.",
                "Convert imperative code to functional style.",
                "Apply the repository pattern to data access.",
                "Refactor to eliminate global state.",
                "Convert synchronous I/O to async operations.",
                "Apply the strategy pattern for algorithm selection.",
            ],
        }
        
        type_questions = questions.get(dtype, questions["bugfixing"])
        return type_questions[index % len(type_questions)]
    
    def _generate_answer(self, rng: random.Random, dtype: str, difficulty: str) -> str:
        """Generate a realistic answer with appropriate depth."""
        base_length = {"easy": 150, "medium": 300, "hard": 500}[difficulty]
        
        intro_phrases = [
            "Here's how to approach this problem:\n\n",
            "Let me explain the solution step by step:\n\n",
            "To address this issue, follow these steps:\n\n",
            "The recommended approach is:\n\n",
        ]
        
        # TESTCASE_GENERATION: Generate proper pytest code that meets contract
        if dtype in ("testcase", "testcase_generation"):
            return self._generate_pytest_answer(rng, difficulty)
        
        code_block = """
```python
def example_solution():
    # Step 1: Validate input
    if not input_data:
        raise ValueError("Input cannot be empty")
    
    # Step 2: Process data
    result = process(input_data)
    
    # Step 3: Return formatted output
    return format_output(result)
```
"""
        
        explanation = """
**Key Points:**
1. Always validate inputs before processing
2. Use proper error handling with specific exceptions
3. Document the expected behavior clearly
4. Write tests to cover edge cases

**Why This Works:**
This approach follows the principle of failing fast - we catch 
issues early before they propagate through the system. The explicit 
error messages help with debugging and the modular structure makes 
the code easier to test and maintain.
"""
        
        if difficulty == "easy":
            return rng.choice(intro_phrases) + code_block.strip()
        elif difficulty == "medium":
            return rng.choice(intro_phrases) + code_block + "\n" + explanation.strip()
        else:
            extra_detail = """
**Advanced Considerations:**
- Consider caching frequently accessed results
- Implement retry logic for transient failures
- Add metrics/logging for observability
- Consider async processing for better scalability

**Common Pitfalls:**
- Not handling edge cases like empty inputs
- Ignoring proper resource cleanup
- Missing timeout configurations
- Inadequate error messages for debugging
"""
            return rng.choice(intro_phrases) + code_block + "\n" + explanation + "\n" + extra_detail.strip()
    
    def _generate_pytest_answer(self, rng: random.Random, difficulty: str) -> str:
        """Generate a proper pytest answer that meets the contract.
        
        Contract requirements:
        - At least one `def test_` function inside a code block
        - At least two `assert` statements
        - At least one of: pytest.mark.parametrize, pytest.raises, or fixture
        """
        intro = rng.choice([
            "Here's a comprehensive test suite:\n\n",
            "Below are the pytest tests covering the requested functionality:\n\n",
            "The following tests validate the expected behavior:\n\n",
        ])
        
        # Select a pytest feature to include
        feature = rng.choice(["parametrize", "raises", "fixture"])
        func_name = rng.choice(["validate_input", "process_data", "calculate_result", "handle_request"])
        
        if feature == "parametrize":
            code = f'''```python
import pytest

@pytest.mark.parametrize("input_val,expected", [
    ("valid_input", True),
    ("another_valid", True),
    ("", False),
    (None, False),
])
def test_{func_name}_with_various_inputs(input_val, expected):
    """Test {func_name} with multiple input scenarios."""
    result = {func_name}(input_val) if input_val else False
    assert result == expected
    assert isinstance(result, bool)
    
def test_{func_name}_returns_correct_type():
    """Test that {func_name} returns the expected type."""
    result = {func_name}("test_value")
    assert result is not None
    assert hasattr(result, '__bool__')
```'''
        elif feature == "raises":
            code = f'''```python
import pytest

def test_{func_name}_raises_on_invalid_input():
    """Test that {func_name} raises ValueError for invalid inputs."""
    with pytest.raises(ValueError, match="Invalid input"):
        {func_name}(None)
    
    with pytest.raises(ValueError):
        {func_name}("")
    
    # Verify error handling doesn't break state
    assert True

def test_{func_name}_success_case():
    """Test {func_name} with valid input."""
    result = {func_name}("valid_input")
    assert result is not None
    assert result == "processed_valid_input"
```'''
        else:  # fixture
            code = f'''```python
import pytest

@pytest.fixture
def sample_data():
    """Provide sample test data."""
    return {{"key": "value", "count": 42}}

@pytest.fixture
def mock_service(mocker):
    """Mock external service dependency."""
    return mocker.Mock(return_value="mocked_response")

def test_{func_name}_with_fixture(sample_data, mock_service):
    """Test {func_name} using fixtures."""
    result = {func_name}(sample_data)
    assert result is not None
    assert "key" in str(result) or result == sample_data
    
def test_{func_name}_handles_empty_data(sample_data):
    """Test edge case with modified fixture data."""
    sample_data["key"] = ""
    result = {func_name}(sample_data)
    assert result is not None
    assert isinstance(result, dict)
```'''
        
        explanation = """
**Test Coverage:**
- Parametrized tests for multiple input scenarios
- Exception handling verification
- Type checking assertions
- Edge case handling
"""
        
        if difficulty == "easy":
            return intro + code
        else:
            return intro + code + "\n\n" + explanation.strip()
    
    def _generate_intents_json(self, rng: random.Random, prompt: str) -> dict[str, Any]:
        """Generate dataset intents JSON."""
        intent_templates = {
            "bugfixing": [
                {"label": "error_diagnosis", "description": "Identify root cause of errors"},
                {"label": "fix_implementation", "description": "Implement correct solution"},
                {"label": "prevention", "description": "Best practices to prevent similar bugs"},
                {"label": "edge_cases", "description": "Handle edge cases and corner cases"},
                {"label": "debugging_strategy", "description": "Systematic debugging approaches"},
            ],
            "testcase_generation": [
                {"label": "boundary_conditions", "description": "Test edge cases and boundary values"},
                {"label": "invalid_inputs", "description": "Handle invalid or malformed inputs"},
                {"label": "parametrization", "description": "Use pytest.mark.parametrize for data-driven tests"},
                {"label": "exception_paths", "description": "Test error handling with pytest.raises"},
                {"label": "stateful_behavior", "description": "Test state changes and side effects"},
                {"label": "mocking", "description": "Mock external dependencies with fixtures"},
                {"label": "async_tests", "description": "Test async functions with pytest-asyncio"},
                {"label": "regression_tests", "description": "Prevent reintroduction of fixed bugs"},
            ],
            "doc_generation": [
                {"label": "api_documentation", "description": "Document API endpoints"},
                {"label": "code_documentation", "description": "Write docstrings and comments"},
                {"label": "architecture_docs", "description": "Describe system architecture"},
                {"label": "user_guides", "description": "Create end-user documentation"},
                {"label": "examples", "description": "Provide usage examples"},
            ],
            "code_review": [
                {"label": "security_review", "description": "Identify security issues"},
                {"label": "performance_review", "description": "Find performance bottlenecks"},
                {"label": "maintainability", "description": "Assess code maintainability"},
                {"label": "best_practices", "description": "Check adherence to standards"},
                {"label": "architecture", "description": "Review design decisions"},
            ],
            "refactoring": [
                {"label": "decomposition", "description": "Break down complex functions"},
                {"label": "abstraction", "description": "Extract reusable components"},
                {"label": "pattern_application", "description": "Apply design patterns"},
                {"label": "modernization", "description": "Update to modern practices"},
                {"label": "simplification", "description": "Reduce complexity"},
            ],
        }
        
        # Find dataset type in prompt
        for dtype, intents in intent_templates.items():
            if dtype.replace("_", "") in prompt.lower().replace("_", "").replace(" ", ""):
                selected = rng.sample(intents, min(len(intents), 4))
                return {"intents": selected}
        
        # Default
        return {"intents": intent_templates["bugfixing"][:4]}
    
    def _generate_critique_json(self, rng: random.Random, prompt: str) -> dict[str, Any]:
        """Generate critique JSON response."""
        # Simulate finding some issues
        num_items = 10
        for word in prompt.split():
            if word.isdigit():
                num_items = int(word)
                break
        
        # Randomly reject a few items (0-20%)
        reject_count = rng.randint(0, max(1, num_items // 5))
        reject_indices = sorted(rng.sample(range(num_items), reject_count)) if reject_count > 0 else []
        
        improvement_notes = []
        if reject_indices:
            improvement_notes.append("Some questions were too similar in structure")
        if rng.random() > 0.5:
            improvement_notes.append("Consider adding more diverse examples")
        if rng.random() > 0.7:
            improvement_notes.append("Some answers could be more detailed")
        
        return {
            "reject_indices": reject_indices,
            "improvement_notes": improvement_notes,
            "quality_assessment": rng.choice(["good", "acceptable", "needs_improvement"]),
            "duplicate_pairs": [],
            "low_quality_indices": [],
        }
    
    def _generate_uniqueness_json(self, rng: random.Random, prompt: str) -> dict[str, Any]:
        """Generate uniqueness evaluation JSON."""
        conceptual_score = rng.randint(60, 90)
        
        return {
            "conceptual_diversity_score": conceptual_score,
            "reasoning": f"The dataset shows {'strong' if conceptual_score > 75 else 'moderate'} conceptual diversity with varied question types and answer approaches.",
            "strengths": [
                "Good coverage of different subtopics",
                "Varied question structures",
                "Answers demonstrate multiple solution approaches",
            ],
            "weaknesses": [
                "Some thematic overlap between items",
            ] if conceptual_score < 80 else [],
        }
    
    def _generate_generic_json(self, rng: random.Random, prompt: str) -> dict[str, Any]:
        """Generate generic JSON response."""
        return {
            "status": "success",
            "message": "Mock response generated for testing",
            "data": {},
        }
    
    def _generate_pytest_qa(
        self, 
        rng: random.Random, 
        index: int, 
        intent_label: str | None, 
        difficulty: str
    ) -> tuple[str, str]:
        """Generate proper pytest test code Q&A pairs.
        
        Each answer MUST include:
        - At least one function with `def test_`
        - At least two `assert` statements
        - At least one of: pytest.mark.parametrize, pytest.raises, or fixture usage
        - Code inside a Python code block
        """
        # Questions by intent
        questions_by_intent = {
            "boundary_conditions": [
                "Write pytest tests to verify boundary conditions for a function that calculates pagination offsets.",
                "Create test cases that check edge values for a discount calculator function.",
                "Write tests for boundary conditions in a list slicing utility.",
                "Test boundary conditions for a date range validator function.",
                "Create pytest tests for boundary values in a rate limiter.",
            ],
            "invalid_inputs": [
                "Write pytest tests for invalid inputs to a user registration validator.",
                "Create tests that verify proper error handling for malformed JSON parsing.",
                "Test invalid input handling for an email validation function.",
                "Write pytest tests for invalid arguments to a file path resolver.",
                "Create test cases for invalid data types passed to a serializer.",
            ],
            "parametrization": [
                "Write parametrized pytest tests for a currency conversion function.",
                "Create data-driven tests for a string formatting utility.",
                "Write parametrized tests covering multiple scenarios for a sorting function.",
                "Create pytest parametrized tests for a password strength checker.",
                "Write parametrized test cases for a URL path normalizer.",
            ],
            "exception_paths": [
                "Write pytest tests using pytest.raises for a database connection handler.",
                "Create tests that verify exception handling in a file upload service.",
                "Write tests for exception paths in an API rate limiter.",
                "Test that proper exceptions are raised for configuration errors.",
                "Write pytest tests for error handling in a payment processor.",
            ],
            "stateful_behavior": [
                "Write pytest tests for stateful behavior in a shopping cart class.",
                "Create tests that verify state transitions in a workflow engine.",
                "Write tests for state management in a user session handler.",
                "Test stateful behavior of a connection pool manager.",
                "Create pytest tests for state changes in a cache invalidation system.",
            ],
            "mocking": [
                "Write pytest tests with mocked external API calls for a weather service.",
                "Create tests using fixtures to mock database operations.",
                "Write pytest tests that mock file system operations.",
                "Test a notification service with mocked email and SMS providers.",
                "Create tests with mocked time for a scheduler component.",
            ],
            "async_tests": [
                "Write async pytest tests for an HTTP client wrapper.",
                "Create async tests for a websocket message handler.",
                "Write pytest-asyncio tests for a background task scheduler.",
                "Test async database operations with proper fixtures.",
                "Create async tests for a message queue consumer.",
            ],
            "regression_tests": [
                "Write regression tests for a bug where null values caused crashes.",
                "Create pytest tests to prevent reintroduction of a timezone conversion bug.",
                "Write regression tests for a race condition fix in concurrent uploads.",
                "Test that the fix for duplicate record creation holds.",
                "Create regression tests for a memory leak fix in a cache manager.",
            ],
        }
        
        # Answers with proper pytest code by intent
        answers_by_intent = {
            "boundary_conditions": [
                '''Here are pytest tests for boundary conditions:

```python
import pytest

@pytest.fixture
def paginator():
    """Fixture to create a paginator instance."""
    from myapp.utils import Paginator
    return Paginator(total_items=100, items_per_page=10)

class TestPaginationBoundaries:
    @pytest.mark.parametrize("page,expected_offset", [
        (1, 0),      # First page
        (10, 90),    # Last page
        (5, 40),     # Middle page
    ])
    def test_offset_calculation(self, paginator, page, expected_offset):
        """Test offset calculation for various page numbers."""
        offset = paginator.get_offset(page)
        assert offset == expected_offset
        assert offset >= 0
    
    def test_first_page_boundary(self, paginator):
        """Test that first page starts at offset 0."""
        offset = paginator.get_offset(1)
        assert offset == 0
        assert paginator.get_items_for_page(1) is not None
    
    def test_last_page_boundary(self, paginator):
        """Test last page returns correct offset."""
        last_page = paginator.total_pages
        offset = paginator.get_offset(last_page)
        assert offset == (last_page - 1) * paginator.items_per_page
        assert offset < paginator.total_items
```''',
                '''Here are tests for boundary conditions in a discount calculator:

```python
import pytest
from decimal import Decimal
from myapp.pricing import DiscountCalculator

@pytest.fixture
def calculator():
    """Fixture providing a fresh calculator instance."""
    return DiscountCalculator()

class TestDiscountBoundaries:
    @pytest.mark.parametrize("price,discount_pct,expected", [
        (100.00, 0, 100.00),    # No discount
        (100.00, 100, 0.00),    # Full discount  
        (100.00, 50, 50.00),    # Half discount
        (0.01, 10, 0.009),      # Minimum price
    ])
    def test_discount_at_boundaries(self, calculator, price, discount_pct, expected):
        """Test discount calculation at boundary values."""
        result = calculator.apply_discount(Decimal(str(price)), discount_pct)
        assert result == pytest.approx(expected, rel=1e-2)
        assert result >= 0
    
    def test_zero_price_boundary(self, calculator):
        """Test that zero price remains zero after any discount."""
        result = calculator.apply_discount(Decimal("0"), 50)
        assert result == Decimal("0")
        assert isinstance(result, Decimal)
```''',
            ],
            "invalid_inputs": [
                '''Here are pytest tests for invalid inputs:

```python
import pytest
from myapp.validators import UserRegistrationValidator

@pytest.fixture
def validator():
    """Fixture for validator instance."""
    return UserRegistrationValidator()

class TestInvalidInputHandling:
    @pytest.mark.parametrize("email", [
        "",
        None,
        "not-an-email",
        "@missing-local.com",
        "missing-domain@",
        "spaces not@allowed.com",
    ])
    def test_invalid_email_rejected(self, validator, email):
        """Test that invalid emails are properly rejected."""
        with pytest.raises(ValueError, match="Invalid email"):
            validator.validate_email(email)
        assert validator.is_valid_email(email) is False
    
    def test_null_username_raises_error(self, validator):
        """Test that null username raises appropriate error."""
        with pytest.raises(TypeError):
            validator.validate_username(None)
        assert validator.last_error is not None
    
    def test_empty_password_rejected(self, validator):
        """Test empty password handling."""
        with pytest.raises(ValueError, match="Password cannot be empty"):
            validator.validate_password("")
        assert validator.validation_passed is False
```''',
            ],
            "parametrization": [
                '''Here are parametrized pytest tests:

```python
import pytest
from myapp.currency import CurrencyConverter

@pytest.fixture
def converter():
    """Fixture providing currency converter with test rates."""
    rates = {"USD": 1.0, "EUR": 0.85, "GBP": 0.73, "JPY": 110.0}
    return CurrencyConverter(rates)

class TestCurrencyConversion:
    @pytest.mark.parametrize("amount,from_curr,to_curr,expected", [
        (100, "USD", "EUR", 85.0),
        (100, "EUR", "USD", 117.65),
        (1000, "USD", "JPY", 110000.0),
        (0, "USD", "EUR", 0.0),
        (1, "GBP", "GBP", 1.0),
    ])
    def test_conversion_accuracy(self, converter, amount, from_curr, to_curr, expected):
        """Test currency conversion with various inputs."""
        result = converter.convert(amount, from_curr, to_curr)
        assert result == pytest.approx(expected, rel=0.01)
        assert result >= 0
    
    @pytest.mark.parametrize("invalid_currency", ["XXX", "", None, "EURO"])
    def test_invalid_currency_raises(self, converter, invalid_currency):
        """Test that invalid currencies raise ValueError."""
        with pytest.raises(ValueError):
            converter.convert(100, "USD", invalid_currency)
        assert converter.last_error is not None
```''',
            ],
            "exception_paths": [
                '''Here are pytest tests for exception handling:

```python
import pytest
from unittest.mock import Mock, patch
from myapp.database import DatabaseConnection, ConnectionError, TimeoutError

@pytest.fixture
def db_config():
    """Fixture providing test database configuration."""
    return {"host": "localhost", "port": 5432, "timeout": 5}

class TestDatabaseExceptions:
    def test_connection_refused_raises_error(self, db_config):
        """Test that connection refused raises ConnectionError."""
        with patch("myapp.database.socket.connect") as mock_connect:
            mock_connect.side_effect = OSError("Connection refused")
            
            with pytest.raises(ConnectionError, match="Failed to connect"):
                db = DatabaseConnection(**db_config)
                db.connect()
            
            assert mock_connect.called
            assert db.is_connected is False
    
    def test_timeout_raises_timeout_error(self, db_config):
        """Test that connection timeout raises TimeoutError."""
        with patch("myapp.database.socket.connect") as mock_connect:
            mock_connect.side_effect = TimeoutError("Connection timed out")
            
            with pytest.raises(TimeoutError):
                db = DatabaseConnection(**db_config)
                db.connect()
            
            assert db.retry_count > 0
            assert not db.is_connected
    
    def test_invalid_credentials_error(self, db_config):
        """Test invalid credentials handling."""
        db_config["password"] = "wrong_password"
        
        with pytest.raises(ConnectionError, match="Authentication failed"):
            db = DatabaseConnection(**db_config)
            db.connect()
        
        assert db.auth_attempts >= 1
```''',
            ],
            "stateful_behavior": [
                '''Here are tests for stateful behavior:

```python
import pytest
from myapp.cart import ShoppingCart, CartItem

@pytest.fixture
def cart():
    """Fixture providing an empty shopping cart."""
    return ShoppingCart()

@pytest.fixture
def sample_items():
    """Fixture providing sample cart items."""
    return [
        CartItem(id="SKU001", name="Widget", price=10.0),
        CartItem(id="SKU002", name="Gadget", price=25.0),
    ]

class TestShoppingCartState:
    def test_add_item_updates_state(self, cart, sample_items):
        """Test that adding items properly updates cart state."""
        cart.add_item(sample_items[0])
        
        assert cart.item_count == 1
        assert cart.total == 10.0
        assert cart.is_empty is False
    
    def test_remove_item_updates_state(self, cart, sample_items):
        """Test state after removing items."""
        cart.add_item(sample_items[0])
        cart.add_item(sample_items[1])
        cart.remove_item("SKU001")
        
        assert cart.item_count == 1
        assert cart.total == 25.0
        assert "SKU001" not in cart.items
    
    def test_clear_resets_state(self, cart, sample_items):
        """Test that clear resets all state."""
        for item in sample_items:
            cart.add_item(item)
        
        cart.clear()
        
        assert cart.is_empty is True
        assert cart.total == 0
        assert cart.item_count == 0
```''',
            ],
            "mocking": [
                '''Here are pytest tests with mocking:

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
from myapp.weather import WeatherService

@pytest.fixture
def mock_api_response():
    """Fixture providing mock API response."""
    return {
        "temperature": 72,
        "conditions": "sunny",
        "humidity": 45
    }

@pytest.fixture
def weather_service():
    """Fixture providing weather service instance."""
    return WeatherService(api_key="test_key")

class TestWeatherServiceMocking:
    def test_get_weather_with_mocked_api(self, weather_service, mock_api_response):
        """Test weather fetching with mocked external API."""
        with patch.object(weather_service, "_call_api") as mock_call:
            mock_call.return_value = mock_api_response
            
            result = weather_service.get_current_weather("New York")
            
            assert result["temperature"] == 72
            assert result["conditions"] == "sunny"
            mock_call.assert_called_once_with("New York")
    
    def test_api_failure_handling(self, weather_service):
        """Test handling of API failures."""
        with patch.object(weather_service, "_call_api") as mock_call:
            mock_call.side_effect = ConnectionError("API unavailable")
            
            with pytest.raises(ConnectionError):
                weather_service.get_current_weather("London")
            
            assert mock_call.called
            assert weather_service.last_error is not None
```''',
            ],
            "async_tests": [
                '''Here are async pytest tests:

```python
import pytest
import asyncio
from myapp.http_client import AsyncHTTPClient

@pytest.fixture
async def http_client():
    """Async fixture providing HTTP client."""
    client = AsyncHTTPClient(base_url="https://api.example.com")
    yield client
    await client.close()

@pytest.fixture
def mock_response():
    """Fixture for mock HTTP response."""
    return {"status": "ok", "data": {"id": 1, "name": "Test"}}

class TestAsyncHTTPClient:
    @pytest.mark.asyncio
    async def test_get_request(self, http_client, mock_response):
        """Test async GET request handling."""
        with pytest.MonkeyPatch.context() as mp:
            async def mock_fetch(*args, **kwargs):
                return mock_response
            
            mp.setattr(http_client, "_fetch", mock_fetch)
            
            result = await http_client.get("/users/1")
            
            assert result["status"] == "ok"
            assert result["data"]["id"] == 1
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, http_client):
        """Test that timeouts are properly handled."""
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                http_client.get("/slow-endpoint"),
                timeout=0.001
            )
        
        assert http_client.timeout_count >= 1
        assert http_client.is_healthy is False
```''',
            ],
            "regression_tests": [
                '''Here are regression tests:

```python
import pytest
from myapp.processor import DataProcessor

@pytest.fixture
def processor():
    """Fixture providing data processor instance."""
    return DataProcessor()

class TestNullValueRegression:
    """Regression tests for bug #1234: Null values caused crashes."""
    
    @pytest.mark.parametrize("data", [
        {"name": None, "value": 10},
        {"name": "test", "value": None},
        {"name": None, "value": None},
        None,
    ])
    def test_null_values_handled_gracefully(self, processor, data):
        """Ensure null values don't cause crashes (regression #1234)."""
        # This used to raise AttributeError before the fix
        result = processor.process(data)
        
        assert result is not None
        assert processor.error_count == 0
    
    def test_null_in_nested_structure(self, processor):
        """Test null handling in nested data structures."""
        data = {"outer": {"inner": None, "values": [1, None, 3]}}
        
        result = processor.process_nested(data)
        
        assert result["processed"] is True
        assert result["null_count"] == 2
    
    def test_empty_collection_handling(self, processor):
        """Ensure empty collections are handled (related to #1234)."""
        for empty_data in [[], {}, "", set()]:
            result = processor.process(empty_data)
            assert result is not None
            assert processor.last_input_type is not None
```''',
            ],
        }
        
        # Select based on intent or use random
        if intent_label and intent_label in questions_by_intent:
            questions = questions_by_intent[intent_label]
            answers = answers_by_intent.get(intent_label, answers_by_intent["boundary_conditions"])
        else:
            # Pick random intent
            all_intents = list(questions_by_intent.keys())
            chosen_intent = all_intents[index % len(all_intents)]
            questions = questions_by_intent[chosen_intent]
            answers = answers_by_intent.get(chosen_intent, answers_by_intent["boundary_conditions"])
        
        question = questions[index % len(questions)]
        answer = answers[index % len(answers)]
        
        return question, answer
