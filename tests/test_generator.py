"""Tests for the DatasetGenerator module."""

import pytest

from distillery.dataset_generator import DatasetGenerator, TemplateGenerator
from distillery.llm.mock import MockLLMClient
from distillery.schemas import GenerationRequest, UserConstraints


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    return MockLLMClient(seed=42)


@pytest.fixture
def generator(mock_llm):
    """Create a generator instance with a fixed seed for reproducibility."""
    return DatasetGenerator(llm_client=mock_llm, seed=42)


@pytest.fixture
def template_generator():
    """Create a template-only generator for testing."""
    return TemplateGenerator(seed=42)


@pytest.fixture
def sample_request():
    """Create a sample generation request."""
    return GenerationRequest(
        prompt="A code assistant for Python debugging",
        dataset_types=["bugfixing", "testcase_generation"],
        qa_per_type=5,
        constraints=UserConstraints(
            tone="technical",
            difficulty="medium",
        ),
        use_llm=False,  # Use template fallback for predictable tests
    )


class TestOutputShape:
    """Tests for generator output structure."""
    
    def test_output_has_project_summary(self, generator, sample_request):
        """Output should include a project summary."""
        result = generator.generate(sample_request)
        
        assert hasattr(result, "project_summary")
        assert isinstance(result.project_summary, str)
        assert len(result.project_summary) > 0
    
    def test_output_has_correct_number_of_datasets(self, generator, sample_request):
        """Output should have one dataset per requested type."""
        result = generator.generate(sample_request)
        
        assert len(result.datasets) == len(sample_request.dataset_types)
    
    def test_dataset_types_match_request(self, generator, sample_request):
        """Dataset types should match the requested types."""
        result = generator.generate(sample_request)
        
        output_types = {ds.type for ds in result.datasets}
        expected_types = set(sample_request.dataset_types)
        
        assert output_types == expected_types
    
    def test_each_dataset_has_items(self, generator, sample_request):
        """Each dataset should have items."""
        result = generator.generate(sample_request)
        
        for dataset in result.datasets:
            assert hasattr(dataset, "items")
            assert isinstance(dataset.items, list)
    
    def test_items_have_required_fields(self, generator, sample_request):
        """Each Q&A item should have question, answer, and metadata."""
        result = generator.generate(sample_request)
        
        for dataset in result.datasets:
            for item in dataset.items:
                assert hasattr(item, "question")
                assert hasattr(item, "answer")
                assert hasattr(item, "metadata")
                assert isinstance(item.question, str)
                assert isinstance(item.answer, str)
                assert isinstance(item.metadata, dict)


class TestItemCounts:
    """Tests for correct item counts."""
    
    def test_correct_items_per_type(self, template_generator):
        """Each dataset should have the requested number of items."""
        request = GenerationRequest(
            prompt="Test prompt",
            dataset_types=["bugfixing"],
            qa_per_type=10,
            constraints=UserConstraints(),
            use_llm=False,
        )
        
        result = template_generator.generate(request)
        
        assert len(result.datasets[0].items) == 10
    
    def test_correct_items_multiple_types(self, template_generator):
        """Multiple datasets should each have correct item counts."""
        request = GenerationRequest(
            prompt="Test prompt",
            dataset_types=["bugfixing", "testcase_generation", "doc_generation"],
            qa_per_type=7,
            constraints=UserConstraints(),
            use_llm=False,
        )
        
        result = template_generator.generate(request)
        
        for dataset in result.datasets:
            assert len(dataset.items) == 7
    
    def test_single_item_request(self, template_generator):
        """Should handle single item request."""
        request = GenerationRequest(
            prompt="Test",
            dataset_types=["bugfixing"],
            qa_per_type=1,
            constraints=UserConstraints(),
            use_llm=False,
        )
        
        result = template_generator.generate(request)
        
        assert len(result.datasets[0].items) == 1
    
    def test_large_item_count(self, template_generator):
        """Should handle larger item counts."""
        request = GenerationRequest(
            prompt="Test",
            dataset_types=["bugfixing"],
            qa_per_type=50,
            constraints=UserConstraints(),
            use_llm=False,
        )
        
        result = template_generator.generate(request)
        
        assert len(result.datasets[0].items) == 50


class TestContentQuality:
    """Tests for content quality and validity."""
    
    def test_questions_are_non_empty(self, generator, sample_request):
        """Questions should not be empty strings."""
        result = generator.generate(sample_request)
        
        for dataset in result.datasets:
            for item in dataset.items:
                assert len(item.question.strip()) > 0
    
    def test_answers_are_non_empty(self, generator, sample_request):
        """Answers should not be empty strings."""
        result = generator.generate(sample_request)
        
        for dataset in result.datasets:
            for item in dataset.items:
                assert len(item.answer.strip()) > 0
    
    def test_answers_have_reasonable_length(self, generator, sample_request):
        """Answers should have reasonable length (not too short)."""
        result = generator.generate(sample_request)
        
        for dataset in result.datasets:
            for item in dataset.items:
                # Answers should be at least 50 characters
                assert len(item.answer) >= 50
    
    def test_metadata_has_id(self, generator, sample_request):
        """Each item's metadata should include an ID."""
        result = generator.generate(sample_request)
        
        for dataset in result.datasets:
            for item in dataset.items:
                assert "id" in item.metadata
    
    def test_metadata_has_intent_label(self, generator, sample_request):
        """Each item's metadata should include intent label (V2)."""
        result = generator.generate(sample_request)
        
        for dataset in result.datasets:
            for item in dataset.items:
                assert "intent_label" in item.metadata


class TestReproducibility:
    """Tests for reproducibility with seeds."""
    
    def test_same_seed_produces_same_output(self):
        """Same seed should produce identical outputs with template generator."""
        request = GenerationRequest(
            prompt="Test",
            dataset_types=["bugfixing"],
            qa_per_type=5,
            constraints=UserConstraints(),
            use_llm=False,  # Use template for reproducibility
        )
        
        gen1 = TemplateGenerator(seed=123)
        gen2 = TemplateGenerator(seed=123)
        
        result1 = gen1.generate(request)
        result2 = gen2.generate(request)
        
        for i, (item1, item2) in enumerate(zip(
            result1.datasets[0].items,
            result2.datasets[0].items,
        )):
            assert item1.question == item2.question
            assert item1.answer == item2.answer
    
    def test_different_seeds_produce_different_output(self):
        """Different seeds should produce different outputs."""
        request = GenerationRequest(
            prompt="Test",
            dataset_types=["bugfixing"],
            qa_per_type=5,
            constraints=UserConstraints(),
            use_llm=False,
        )
        
        gen1 = TemplateGenerator(seed=123)
        gen2 = TemplateGenerator(seed=456)
        
        result1 = gen1.generate(request)
        result2 = gen2.generate(request)
        
        # At least some items should differ
        different_count = sum(
            1 for item1, item2 in zip(
                result1.datasets[0].items,
                result2.datasets[0].items,
            )
            if item1.question != item2.question
        )
        
        assert different_count > 0


class TestUnknownDatasetTypes:
    """Tests for handling unknown dataset types."""
    
    def test_unknown_type_uses_default_templates(self, template_generator):
        """Unknown dataset types should use default templates."""
        request = GenerationRequest(
            prompt="Test",
            dataset_types=["custom_unknown_type"],
            qa_per_type=3,
            constraints=UserConstraints(),
            use_llm=False,
        )
        
        result = template_generator.generate(request)
        
        assert len(result.datasets) == 1
        assert result.datasets[0].type == "custom_unknown_type"
        assert len(result.datasets[0].items) == 3


class TestV2Features:
    """Tests for V2 LLM-backed generation features."""
    
    def test_output_includes_generation_method(self, generator, sample_request):
        """V2 output should include generation method."""
        result = generator.generate(sample_request)
        
        assert hasattr(result, "generation_method")
        assert result.generation_method in ["llm", "template"]
    
    def test_output_includes_intents(self, generator, sample_request):
        """V2 datasets should include intents."""
        result = generator.generate(sample_request)
        
        for dataset in result.datasets:
            assert hasattr(dataset, "intents")
            assert isinstance(dataset.intents, list)
    
    def test_metadata_includes_v2_fields(self, generator, sample_request):
        """V2 metadata should include difficulty and training value."""
        result = generator.generate(sample_request)

        for dataset in result.datasets:
            for item in dataset.items:
                assert "difficulty" in item.metadata
                assert "estimated_training_value" in item.metadata
                assert "source" in item.metadata


class TestSelfContainedTestcasePrompt:
    """The testcase prompt must instruct the model to emit runnable, self-contained code."""

    def test_prompt_requires_self_contained_code(self, generator):
        from distillery.schemas import DatasetIntent

        intent = DatasetIntent(label="boundary_conditions", description="edge cases")
        prompt = generator._get_testcase_generation_prompt(
            count=3, intent=intent, difficulty="medium", tone="technical", existing_context=""
        )
        assert "SELF-CONTAINED" in prompt
        # The example must define the implementation in-block (no undefined SomeClass).
        assert "class Calculator" in prompt
        assert "SomeClass" not in prompt
