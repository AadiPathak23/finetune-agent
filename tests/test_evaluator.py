"""Tests for the Evaluator module."""

import pytest

from finetune_agent.evaluator import Evaluator
from finetune_agent.llm.mock import MockLLMClient
from finetune_agent.schemas import Dataset, DatasetOutput, QAPair


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    return MockLLMClient(seed=42)


@pytest.fixture
def evaluator(mock_llm):
    """Create an evaluator instance with mock LLM."""
    return Evaluator(llm_client=mock_llm)


class TestUniquenessScoring:
    """Tests for uniqueness scoring functionality."""
    
    def test_uniqueness_score_returns_value_in_range(self, evaluator):
        """Uniqueness score should be between 0 and 100."""
        texts = [
            "How do I fix a null pointer exception?",
            "What causes memory leaks in Python?",
            "Why is my async function not awaited?",
            "How to handle file not found errors?",
            "What is the best way to validate user input?",
        ]
        
        overall, lexical, structural, conceptual = evaluator.calculate_uniqueness_score(texts)
        
        assert 0 <= overall <= 100
        assert 0 <= lexical <= 100
        assert 0 <= structural <= 100
        assert 0 <= conceptual <= 100
    
    def test_identical_texts_have_low_uniqueness(self, evaluator):
        """Identical texts should have low uniqueness score."""
        texts = [
            "How do I fix a bug?",
            "How do I fix a bug?",
            "How do I fix a bug?",
            "How do I fix a bug?",
        ]
        
        overall, lexical, structural, conceptual = evaluator.calculate_uniqueness_score(texts)
        
        # Identical texts should have low uniqueness
        assert overall < 50
    
    def test_diverse_texts_have_high_uniqueness(self, evaluator):
        """Diverse texts should have high uniqueness score."""
        texts = [
            "How do I implement a binary search tree in Python?",
            "What are the best practices for securing REST APIs?",
            "Explain the difference between composition and inheritance.",
            "Write a function to calculate Fibonacci numbers using memoization.",
            "How does garbage collection work in JavaScript?",
            "What is the observer design pattern and when to use it?",
            "Describe the process of database normalization.",
            "How to implement caching in a distributed system?",
        ]
        
        overall, lexical, structural, conceptual = evaluator.calculate_uniqueness_score(texts)
        
        # Diverse texts should have high uniqueness
        assert overall > 60
    
    def test_single_text_has_perfect_uniqueness(self, evaluator):
        """A single text should have perfect uniqueness."""
        texts = ["How do I fix a bug in my code?"]
        
        overall, lexical, structural, conceptual = evaluator.calculate_uniqueness_score(texts)
        
        assert overall == 100.0
    
    def test_empty_texts_return_zero(self, evaluator):
        """Empty text list should return 0."""
        overall, lexical, structural, conceptual = evaluator.calculate_uniqueness_score([])
        
        assert overall == 0.0
    
    def test_partial_overlap_has_moderate_uniqueness(self, evaluator):
        """Texts with partial overlap should have moderate uniqueness."""
        texts = [
            "How do I fix a bug in Python code?",
            "How do I fix a bug in JavaScript code?",
            "How do I fix a bug in TypeScript code?",
            "How do I fix a bug in Go code?",
        ]
        
        overall, lexical, structural, conceptual = evaluator.calculate_uniqueness_score(texts)
        
        # Partial overlap should give moderate scores
        assert 30 <= overall <= 80


class TestFullEvaluation:
    """Tests for the complete evaluation pipeline."""
    
    def test_evaluate_returns_correct_structure(self, evaluator):
        """Evaluation should return all required fields."""
        dataset = DatasetOutput(
            project_summary="Test summary",
            datasets=[
                Dataset(
                    type="bugfixing",
                    items=[
                        QAPair(
                            question="How to fix null pointer?",
                            answer="Check for null before accessing.",
                            metadata={"id": "1"},
                        ),
                        QAPair(
                            question="What causes stack overflow?",
                            answer="Infinite recursion or large allocations.",
                            metadata={"id": "2"},
                        ),
                    ],
                ),
            ],
        )
        
        result = evaluator.evaluate(dataset)
        
        assert hasattr(result, "dataset_evaluations")
        assert hasattr(result, "overall_rating")
        assert hasattr(result, "feedback")
        assert len(result.dataset_evaluations) == 1
        assert 0 <= result.overall_rating <= 100
        assert len(result.feedback) > 0
    
    def test_evaluate_handles_multiple_datasets(self, evaluator):
        """Evaluation should handle multiple dataset types."""
        dataset = DatasetOutput(
            project_summary="Test summary",
            datasets=[
                Dataset(
                    type="bugfixing",
                    items=[
                        QAPair(question="Q1", answer="A1" * 20, metadata={}),
                        QAPair(question="Q2", answer="A2" * 20, metadata={}),
                    ],
                ),
                Dataset(
                    type="testing",
                    items=[
                        QAPair(question="Q3", answer="A3" * 20, metadata={}),
                        QAPair(question="Q4", answer="A4" * 20, metadata={}),
                    ],
                ),
            ],
        )
        
        result = evaluator.evaluate(dataset)
        
        assert len(result.dataset_evaluations) == 2
        assert result.dataset_evaluations[0].dataset_type == "bugfixing"
        assert result.dataset_evaluations[1].dataset_type == "testing"
    
    def test_overall_rating_in_valid_range(self, evaluator):
        """Overall rating should be between 0 and 100."""
        dataset = DatasetOutput(
            project_summary="Test",
            datasets=[
                Dataset(
                    type="test",
                    items=[
                        QAPair(
                            question="Q" * 50,
                            answer="A" * 200,
                            metadata={},
                        )
                        for _ in range(5)
                    ],
                ),
            ],
        )
        
        result = evaluator.evaluate(dataset)
        
        assert 0 <= result.overall_rating <= 100


class TestLexicalDiversity:
    """Tests for lexical diversity calculation."""
    
    def test_lexical_diversity_with_varied_vocabulary(self, evaluator):
        """High vocabulary variety should yield higher diversity."""
        high_variety = [
            "The quick brown fox jumps over the lazy dog.",
            "Programming languages include Python, Java, and Rust.",
            "Machine learning algorithms can classify and predict.",
        ]
        
        low_variety = [
            "The the the the the the the.",
            "The the the and the the the.",
            "The the the or the the the.",
        ]
        
        high_score = evaluator._calculate_lexical_diversity(high_variety)
        low_score = evaluator._calculate_lexical_diversity(low_variety)
        
        assert high_score > low_score


class TestNgramUniqueness:
    """Tests for n-gram uniqueness calculation."""
    
    def test_ngram_uniqueness_with_unique_texts(self, evaluator):
        """Completely unique texts should have high n-gram uniqueness."""
        unique_texts = [
            "Implement binary search algorithm efficiently.",
            "Design patterns improve code maintainability.",
            "Database transactions ensure consistency.",
        ]
        
        score = evaluator._calculate_ngram_uniqueness(unique_texts)
        
        assert score > 70
    
    def test_ngram_uniqueness_with_repeated_phrases(self, evaluator):
        """Repeated phrases should lower n-gram uniqueness."""
        repeated_texts = [
            "How to implement this feature?",
            "How to implement that feature?",
            "How to implement another feature?",
        ]
        
        score = evaluator._calculate_ngram_uniqueness(repeated_texts)
        
        # Should be lower due to repeated "How to implement"
        assert score < 90


class TestConceptualScoring:
    """Tests for V2 LLM-assisted conceptual scoring."""
    
    def test_conceptual_score_returns_value_in_range(self, evaluator):
        """Conceptual score should be between 0 and 100."""
        texts = [
            "How to debug memory leaks?",
            "What is dependency injection?",
            "Explain the SOLID principles.",
        ]
        
        score = evaluator.calculate_conceptual_score(texts)
        
        assert 0 <= score <= 100
    
    def test_conceptual_score_single_text(self, evaluator):
        """Single text should have perfect conceptual score."""
        texts = ["How to debug memory leaks?"]
        
        score = evaluator.calculate_conceptual_score(texts)
        
        assert score == 100.0
    
    def test_conceptual_score_empty_list(self, evaluator):
        """Empty list should have zero conceptual score."""
        score = evaluator.calculate_conceptual_score([])
        
        assert score == 0.0


class TestHealthMetrics:
    """Tests for V2 health metrics calculation."""
    
    def test_health_metrics_structure(self, evaluator):
        """Health metrics should have all required fields."""
        dataset = DatasetOutput(
            project_summary="Test",
            datasets=[
                Dataset(
                    type="bugfixing",
                    items=[
                        QAPair(
                            question="Q1?",
                            answer="A1 " * 50 + "```python\ncode\n```",
                            metadata={"difficulty": "easy", "intent_label": "fix"},
                        ),
                        QAPair(
                            question="Q2?",
                            answer="A2 " * 100,
                            metadata={"difficulty": "medium", "intent_label": "debug"},
                        ),
                    ],
                ),
            ],
        )
        
        metrics = evaluator.calculate_health_metrics(dataset)
        
        assert hasattr(metrics, "avg_answer_length")
        assert hasattr(metrics, "difficulty_distribution")
        assert hasattr(metrics, "intent_coverage_score")
        assert hasattr(metrics, "items_with_code")
        assert hasattr(metrics, "items_with_code_pct")
    
    def test_health_metrics_counts_code(self, evaluator):
        """Health metrics should correctly count items with code."""
        dataset = DatasetOutput(
            project_summary="Test",
            datasets=[
                Dataset(
                    type="test",
                    items=[
                        QAPair(question="Q1", answer="```python\ncode\n```", metadata={}),
                        QAPair(question="Q2", answer="def foo(): pass", metadata={}),
                        QAPair(question="Q3", answer="No code here", metadata={}),
                    ],
                ),
            ],
        )
        
        metrics = evaluator.calculate_health_metrics(dataset)
        
        assert metrics.items_with_code == 2
        assert metrics.items_with_code_pct == pytest.approx(66.67, rel=0.1)
    
    def test_health_metrics_difficulty_distribution(self, evaluator):
        """Health metrics should track difficulty distribution."""
        dataset = DatasetOutput(
            project_summary="Test",
            datasets=[
                Dataset(
                    type="test",
                    items=[
                        QAPair(question="Q1", answer="A1", metadata={"difficulty": "easy"}),
                        QAPair(question="Q2", answer="A2", metadata={"difficulty": "easy"}),
                        QAPair(question="Q3", answer="A3", metadata={"difficulty": "hard"}),
                    ],
                ),
            ],
        )
        
        metrics = evaluator.calculate_health_metrics(dataset)
        
        assert metrics.difficulty_distribution.get("easy") == 2
        assert metrics.difficulty_distribution.get("hard") == 1


class TestV2EvaluationOutput:
    """Tests for V2 evaluation output structure."""
    
    def test_evaluation_includes_component_scores(self, evaluator):
        """V2 evaluation should include lexical, structural, and conceptual scores."""
        dataset = DatasetOutput(
            project_summary="Test",
            datasets=[
                Dataset(
                    type="bugfixing",
                    items=[
                        QAPair(
                            question="How to fix a bug?",
                            answer="Check the logs and debug step by step.",
                            metadata={"difficulty": "medium", "intent_label": "fix"},
                        ),
                    ],
                ),
            ],
        )
        
        result = evaluator.evaluate(dataset)
        
        eval_result = result.dataset_evaluations[0]
        assert hasattr(eval_result, "lexical_score")
        assert hasattr(eval_result, "structural_score")
        assert hasattr(eval_result, "conceptual_score")
    
    def test_evaluation_includes_health_metrics(self, evaluator):
        """V2 evaluation should include health metrics."""
        dataset = DatasetOutput(
            project_summary="Test",
            datasets=[
                Dataset(
                    type="test",
                    items=[
                        QAPair(question="Q?", answer="A" * 100, metadata={}),
                    ],
                ),
            ],
        )
        
        result = evaluator.evaluate(dataset)
        
        assert hasattr(result, "health_metrics")
        assert result.health_metrics is not None
    
    def test_evaluation_includes_llm_feedback(self, evaluator):
        """V2 evaluation should include LLM feedback."""
        dataset = DatasetOutput(
            project_summary="Test",
            datasets=[
                Dataset(
                    type="test",
                    items=[
                        QAPair(question="Q?", answer="A" * 100, metadata={}),
                    ],
                ),
            ],
        )
        
        result = evaluator.evaluate(dataset)
        
        assert hasattr(result, "llm_feedback")
        # LLM feedback may be empty string but should exist
        assert result.llm_feedback is not None
    
    def test_evaluation_includes_warnings(self, evaluator):
        """V2 evaluation should include warnings list."""
        dataset = DatasetOutput(
            project_summary="Test",
            datasets=[
                Dataset(
                    type="test",
                    items=[
                        QAPair(question="Q?", answer="A", metadata={}),  # Very short
                    ],
                ),
            ],
        )
        
        result = evaluator.evaluate(dataset)
        
        assert hasattr(result, "warnings")
        assert isinstance(result.warnings, list)
