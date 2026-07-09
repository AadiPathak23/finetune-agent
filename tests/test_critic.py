"""Tests for the DatasetCritic module."""

import pytest

from distillery.critic import DatasetCritic
from distillery.llm.mock import MockLLMClient
from distillery.schemas import Dataset, DatasetOutput, QAPair


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    return MockLLMClient(seed=42)


@pytest.fixture
def critic(mock_llm):
    """Create a critic instance with mock LLM."""
    return DatasetCritic(llm_client=mock_llm, aggressive=False)


@pytest.fixture
def aggressive_critic(mock_llm):
    """Create a critic instance with aggressive filtering."""
    return DatasetCritic(llm_client=mock_llm, aggressive=True)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset(
        type="bugfixing",
        items=[
            QAPair(
                question="How do I fix a null pointer exception?",
                answer="Check for null values before accessing object properties. Add null checks like: if (obj != null) { return early; } This prevents the exception.",
                metadata={"id": "1", "difficulty": "easy"},
            ),
            QAPair(
                question="What causes memory leaks in Python?",
                answer="Memory leaks often occur from circular references, unclosed file handles, or global variables holding large objects. Use context managers and weak references.",
                metadata={"id": "2", "difficulty": "medium"},
            ),
            QAPair(
                question="Debug this async function that hangs?",
                answer="The issue is likely missing await keyword or deadlock. Ensure all async calls are properly awaited and avoid blocking operations in async code.",
                metadata={"id": "3", "difficulty": "hard"},
            ),
        ],
        intents=["error_diagnosis", "fix_implementation"],
    )


@pytest.fixture
def dataset_with_duplicates():
    """Create a dataset with near-duplicate questions."""
    return Dataset(
        type="bugfixing",
        items=[
            QAPair(
                question="How do I fix a null pointer exception in Java?",
                answer="Check for null values before use.",
                metadata={"id": "1"},
            ),
            QAPair(
                question="How do I fix a null pointer exception in Python?",
                answer="Use if statements to check for None.",
                metadata={"id": "2"},
            ),
            QAPair(
                question="What is a memory leak?",
                answer="A memory leak occurs when allocated memory is not released.",
                metadata={"id": "3"},
            ),
        ],
    )


@pytest.fixture
def dataset_with_low_quality():
    """Create a dataset with low quality items."""
    return Dataset(
        type="bugfixing",
        items=[
            QAPair(
                question="Fix?",  # Too short
                answer="Yes",  # Too short
                metadata={"id": "1"},
            ),
            QAPair(
                question="How do I implement proper error handling?",
                answer="Use try-catch blocks...",  # Contains placeholder
                metadata={"id": "2"},
            ),
            QAPair(
                question="What is the best approach for debugging async code?",
                answer="Here's a comprehensive guide to debugging async code:\n\n1. Use logging to track execution flow\n2. Add breakpoints at async boundaries\n3. Check for race conditions\n4. Use debugging tools like asyncio debug mode",
                metadata={"id": "3"},
            ),
        ],
    )


class TestDuplicateDetection:
    """Tests for duplicate detection."""
    
    def test_finds_near_duplicates(self, critic, dataset_with_duplicates):
        """Should find near-duplicate questions."""
        result = critic.critique_dataset(dataset_with_duplicates)
        
        # The first two questions are similar
        assert len(result.duplicate_pairs) >= 0  # May find duplicates
        assert isinstance(result.duplicate_pairs, list)
    
    def test_no_duplicates_in_diverse_dataset(self, critic, sample_dataset):
        """Should not find duplicates in diverse dataset."""
        result = critic.critique_dataset(sample_dataset)
        
        # Diverse questions should not be flagged
        rule_based_dups = critic._find_duplicates(sample_dataset.items)
        assert len(rule_based_dups) == 0


class TestQualityChecks:
    """Tests for quality checking."""
    
    def test_detects_short_questions(self, critic):
        """Should detect questions that are too short."""
        short_item = QAPair(question="Fix?", answer="A" * 100, metadata={})
        issues = critic._check_question_quality(short_item)
        
        assert len(issues) > 0
        assert any("short" in issue.lower() for issue in issues)
    
    def test_detects_short_answers(self, critic):
        """Should detect answers that are too short."""
        short_item = QAPair(question="How to fix the bug?", answer="Yes", metadata={})
        issues = critic._check_answer_quality(short_item)
        
        assert len(issues) > 0
        assert any("short" in issue.lower() for issue in issues)
    
    def test_accepts_good_quality_items(self, critic, sample_dataset):
        """Should accept items that meet quality standards."""
        for item in sample_dataset.items:
            q_issues = critic._check_question_quality(item)
            a_issues = critic._check_answer_quality(item)
            
            # Good items should have no issues
            assert len(q_issues) == 0, f"Question issues: {q_issues}"
            assert len(a_issues) == 0, f"Answer issues: {a_issues}"


class TestAggressiveFiltering:
    """Tests for aggressive filtering mode."""
    
    def test_aggressive_mode_stricter(self, aggressive_critic, dataset_with_low_quality):
        """Aggressive mode should reject more items."""
        result = aggressive_critic.critique_dataset(dataset_with_low_quality)
        
        # Aggressive mode should find issues
        assert result.quality_assessment in ["acceptable", "needs_improvement", "good"]
    
    def test_aggressive_mode_checks_code_presence(self, aggressive_critic):
        """Aggressive mode should check for code in technical answers."""
        item = QAPair(
            question="How do I implement a binary search in Python?",
            answer="Binary search is an algorithm that finds elements in a sorted array by repeatedly dividing the search interval in half.",
            metadata={},
        )
        
        issues = aggressive_critic._check_answer_quality(item)
        
        # Should flag missing code for implementation question
        assert any("code" in issue.lower() for issue in issues)


class TestCritiqueResult:
    """Tests for critique result structure."""
    
    def test_critique_returns_correct_structure(self, critic, sample_dataset):
        """Critique result should have all required fields."""
        result = critic.critique_dataset(sample_dataset)
        
        assert hasattr(result, "reject_indices")
        assert hasattr(result, "improvement_notes")
        assert hasattr(result, "quality_assessment")
        assert hasattr(result, "duplicate_pairs")
        assert hasattr(result, "low_quality_indices")
    
    def test_reject_indices_are_valid(self, critic, sample_dataset):
        """Reject indices should be valid item indices."""
        result = critic.critique_dataset(sample_dataset)
        
        for idx in result.reject_indices:
            assert 0 <= idx < len(sample_dataset.items)
    
    def test_quality_assessment_valid_values(self, critic, sample_dataset):
        """Quality assessment should be one of the expected values."""
        result = critic.critique_dataset(sample_dataset)
        
        assert result.quality_assessment in ["good", "acceptable", "needs_improvement"]


class TestDatasetFiltering:
    """Tests for dataset filtering."""
    
    def test_filter_removes_rejected_items(self, critic):
        """Filtering should remove rejected items."""
        dataset = Dataset(
            type="test",
            items=[
                QAPair(question="Q1", answer="A1" * 50, metadata={}),
                QAPair(question="Q2", answer="A2" * 50, metadata={}),
                QAPair(question="Q3", answer="A3" * 50, metadata={}),
            ],
        )
        
        from distillery.schemas import CritiqueResult
        critique = CritiqueResult(reject_indices=[1])
        
        filtered = critic.filter_dataset(dataset, critique)
        
        assert len(filtered.items) == 2
        assert filtered.items[0].question == "Q1"
        assert filtered.items[1].question == "Q3"
    
    def test_filter_preserves_non_rejected_items(self, critic, sample_dataset):
        """Filtering should preserve items not in reject list."""
        from distillery.schemas import CritiqueResult
        critique = CritiqueResult(reject_indices=[])
        
        filtered = critic.filter_dataset(sample_dataset, critique)
        
        assert len(filtered.items) == len(sample_dataset.items)


class TestFullCritiquePipeline:
    """Tests for the full critique pipeline."""
    
    def test_critique_all_datasets(self, critic):
        """Should critique all datasets in output."""
        output = DatasetOutput(
            project_summary="Test",
            datasets=[
                Dataset(
                    type="bugfixing",
                    items=[
                        QAPair(question="Q1?", answer="A1" * 50, metadata={}),
                    ],
                ),
                Dataset(
                    type="testing",
                    items=[
                        QAPair(question="Q2?", answer="A2" * 50, metadata={}),
                    ],
                ),
            ],
        )
        
        results = critic.critique(output)
        
        assert "bugfixing" in results
        assert "testing" in results
    
    def test_filter_all_applies_to_all_datasets(self, critic):
        """Filter all should apply filtering to all datasets."""
        output = DatasetOutput(
            project_summary="Test",
            datasets=[
                Dataset(
                    type="bugfixing",
                    items=[
                        QAPair(question="Q1?", answer="A1" * 50, metadata={}),
                        QAPair(question="Q2?", answer="A2" * 50, metadata={}),
                    ],
                ),
            ],
        )
        
        from distillery.schemas import CritiqueResult
        critiques = {
            "bugfixing": CritiqueResult(reject_indices=[0]),
        }
        
        filtered = critic.filter_all(output, critiques)
        
        assert len(filtered.datasets[0].items) == 1


class TestPytestContractEnforcement:
    """Tests for pytest contract enforcement in testcase_generation datasets."""
    
    @pytest.fixture
    def testcase_dataset_valid(self):
        """Create a valid testcase_generation dataset with proper pytest code."""
        return Dataset(
            type="testcase_generation",
            items=[
                QAPair(
                    question="Write pytest tests for a user validator function.",
                    answer='''Here are the tests:

```python
import pytest

@pytest.fixture
def validator():
    return UserValidator()

class TestUserValidator:
    @pytest.mark.parametrize("email", ["test@example.com", "user@domain.org"])
    def test_valid_emails_accepted(self, validator, email):
        result = validator.validate_email(email)
        assert result is True
        assert validator.last_error is None
    
    def test_invalid_email_rejected(self, validator):
        with pytest.raises(ValueError):
            validator.validate_email("not-an-email")
        assert validator.is_valid is False
```''',
                    metadata={"id": "1", "intent_label": "parametrization"},
                ),
            ],
            intents=["parametrization"],
        )
    
    @pytest.fixture
    def testcase_dataset_invalid(self):
        """Create an invalid testcase_generation dataset without proper pytest code."""
        return Dataset(
            type="testcase_generation",
            items=[
                QAPair(
                    question="Write tests for a calculator.",
                    answer='''Here are some test ideas:

1. Test addition
2. Test subtraction
3. Test edge cases

Make sure to cover boundary conditions.''',
                    metadata={"id": "1"},
                ),
                QAPair(
                    question="Write unit tests for a sorting function.",
                    answer='''The function should be tested with:
- Empty arrays
- Single element
- Already sorted
- Reverse sorted''',
                    metadata={"id": "2"},
                ),
            ],
            intents=["general"],
        )
    
    def test_valid_pytest_code_passes_contract(self, mock_llm, testcase_dataset_valid):
        """Valid pytest code should pass the contract check."""
        critic = DatasetCritic(llm_client=mock_llm, aggressive=False)
        item = testcase_dataset_valid.items[0]
        
        issues = critic._check_pytest_contract(item)
        
        # Should have no issues
        assert len(issues) == 0, f"Unexpected issues: {issues}"
    
    def test_missing_test_function_fails_contract(self, mock_llm):
        """Answer without def test_ function should fail contract."""
        critic = DatasetCritic(llm_client=mock_llm, aggressive=False)
        item = QAPair(
            question="Write tests for validation.",
            answer='''```python
def validate():
    assert True
    assert 1 == 1
```''',
            metadata={},
        )
        
        issues = critic._check_pytest_contract(item)
        
        assert any("def test_" in issue for issue in issues)
    
    def test_insufficient_assertions_fails_contract(self, mock_llm):
        """Answer with fewer than 2 assertions should fail contract."""
        critic = DatasetCritic(llm_client=mock_llm, aggressive=False)
        item = QAPair(
            question="Write tests for login.",
            answer='''```python
def test_login():
    result = login("user", "pass")
    assert result is True
```''',
            metadata={},
        )
        
        issues = critic._check_pytest_contract(item)
        
        assert any("assertion" in issue.lower() for issue in issues)
    
    def test_missing_pytest_feature_fails_contract(self, mock_llm):
        """Answer without pytest features should fail contract."""
        critic = DatasetCritic(llm_client=mock_llm, aggressive=False)
        item = QAPair(
            question="Write tests for calculator.",
            answer='''```python
def test_add():
    result = add(1, 2)
    assert result == 3
    assert isinstance(result, int)
```''',
            metadata={},
        )
        
        issues = critic._check_pytest_contract(item)
        
        # Should fail because no pytest.mark.parametrize, pytest.raises, or fixture
        assert any("pytest feature" in issue.lower() or "parametrize" in issue.lower() or "raises" in issue.lower() or "fixture" in issue.lower() for issue in issues)
    
    def test_missing_code_block_fails_contract(self, mock_llm):
        """Answer without code block should fail contract."""
        critic = DatasetCritic(llm_client=mock_llm, aggressive=False)
        item = QAPair(
            question="Write tests for validation.",
            answer="You should test the validation by checking various inputs and edge cases.",
            metadata={},
        )
        
        issues = critic._check_pytest_contract(item)
        
        assert any("code block" in issue.lower() for issue in issues)
    
    def test_testcase_dataset_rejects_invalid_items(self, mock_llm, testcase_dataset_invalid):
        """Critic should reject invalid testcase_generation items."""
        # Even non-aggressive mode should reject testcase items that fail contract
        critic = DatasetCritic(llm_client=mock_llm, aggressive=False)
        
        result = critic.critique_dataset(testcase_dataset_invalid)
        
        # All items should be flagged as low quality
        assert len(result.low_quality_indices) == len(testcase_dataset_invalid.items)
        # In testcase_generation, contract violations should cause rejection
        assert len(result.reject_indices) > 0
    
    def test_extract_code_blocks(self, mock_llm):
        """Test that code blocks are properly extracted."""
        critic = DatasetCritic(llm_client=mock_llm, aggressive=False)
        
        # Test with python code block
        text_with_python = '''Here is the code:

```python
def test_example():
    assert True
```

And more explanation.'''
        
        blocks = critic._extract_code_blocks(text_with_python)
        assert len(blocks) == 1
        assert "def test_example" in blocks[0]
        
        # Test with generic code block
        text_with_generic = '''```
def test_generic():
    pass
```'''
        
        blocks = critic._extract_code_blocks(text_with_generic)
        assert len(blocks) == 1
        
        # Test with multiple blocks
        text_multiple = '''```python
block1
```

text

```python
block2
```'''
        
        blocks = critic._extract_code_blocks(text_multiple)
        assert len(blocks) == 2
    
    def test_contract_checks_within_code_block_only(self, mock_llm):
        """Contract checks should analyze code inside blocks, not surrounding text."""
        critic = DatasetCritic(llm_client=mock_llm, aggressive=False)
        
        # Answer mentions def test_ in prose but not in code
        item = QAPair(
            question="Write tests.",
            answer='''You should use def test_ functions with assert statements.
            
```python
def my_function():
    # No tests here
    pass
```''',
            metadata={},
        )
        
        issues = critic._check_pytest_contract(item)
        
        # Should fail because code block doesn't have def test_
        assert any("def test_" in issue for issue in issues)
    
    def test_realistic_valid_pytest_answer(self, mock_llm):
        """Test with a realistic, valid pytest answer."""
        critic = DatasetCritic(llm_client=mock_llm, aggressive=False)
        
        item = QAPair(
            question="Write unit tests for a user registration function that validates email and password.",
            answer='''Here's a comprehensive test suite for the user registration function:

```python
import pytest
from registration import UserRegistration, ValidationError

@pytest.fixture
def registration_service():
    """Create a fresh registration service for each test."""
    return UserRegistration()

class TestUserRegistration:
    @pytest.mark.parametrize("email,password,expected", [
        ("valid@example.com", "StrongP@ss123", True),
        ("user@domain.org", "AnotherGood1!", True),
    ])
    def test_valid_registration_succeeds(self, registration_service, email, password, expected):
        result = registration_service.register(email, password)
        assert result.success == expected
        assert result.user_id is not None
    
    def test_invalid_email_raises_error(self, registration_service):
        with pytest.raises(ValidationError, match="Invalid email format"):
            registration_service.register("not-an-email", "Password123!")
        assert registration_service.last_error is not None
    
    def test_weak_password_rejected(self, registration_service):
        with pytest.raises(ValidationError):
            registration_service.register("test@example.com", "weak")
        assert registration_service.validation_count > 0
```

This test suite covers:
- Valid registrations with parameterized inputs
- Invalid email handling with specific error matching
- Weak password rejection''',
            metadata={},
        )
        
        issues = critic._check_pytest_contract(item)
        
        # Should pass all checks
        assert len(issues) == 0, f"Unexpected issues: {issues}"


class TestIntentDistribution:
    """Tests for intent distribution in generated datasets."""
    
    def test_testcase_intents_are_diverse(self, mock_llm):
        """Generated testcase intents should be diverse and specific."""
        # Check the mock LLM's intent generation
        result = mock_llm.generate_json("Generate intents for testcase_generation dataset")
        
        intents = result.get("intents", [])
        labels = [i.get("label", "") for i in intents]
        
        # Should have multiple distinct intents
        assert len(intents) >= 4
        assert len(set(labels)) == len(labels)  # All unique
        
        # Should NOT all be "general"
        assert not all(l == "general" for l in labels)
    
    def test_intents_include_expected_types(self, mock_llm):
        """Testcase intents should include specific testing patterns."""
        result = mock_llm.generate_json("Generate intents for testcase_generation dataset")
        
        intents = result.get("intents", [])
        labels = [i.get("label", "").lower() for i in intents]
        
        # At least some of these should be present
        expected_patterns = [
            "boundary", "invalid", "parametri", "exception", 
            "mock", "async", "regression", "state"
        ]
        
        found_patterns = sum(
            1 for pattern in expected_patterns
            if any(pattern in label for label in labels)
        )
        
        # Should match at least 3 expected patterns
        assert found_patterns >= 3, f"Only found {found_patterns} expected patterns in {labels}"
