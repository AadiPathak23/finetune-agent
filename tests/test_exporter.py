"""Tests for the exporter module."""

import json
import tempfile
from pathlib import Path

import pytest

from finetune_agent.exporter import (
    export_all_formats,
    export_chat_jsonl,
    export_instruct_jsonl,
    export_qa_jsonl,
    sample_golden_set,
)
from finetune_agent.schemas import Dataset, DatasetOutput, QAPair


@pytest.fixture
def sample_dataset_output():
    """Create a sample dataset output for testing."""
    return DatasetOutput(
        project_summary="Test project for export testing",
        datasets=[
            Dataset(
                type="bugfixing",
                items=[
                    QAPair(
                        question="How do I fix a null pointer exception?",
                        answer="Check for null before accessing: ```python\nif obj is not None:\n    obj.method()\n```",
                        metadata={
                            "id": "bug_001",
                            "difficulty": "easy",
                            "intent_label": "null_handling",
                            "estimated_training_value": "high",
                            "source": "synthetic",
                        },
                    ),
                    QAPair(
                        question="What causes a memory leak in Python?",
                        answer="Memory leaks occur when objects are not properly dereferenced. Use weak references or ensure cleanup.",
                        metadata={
                            "id": "bug_002",
                            "difficulty": "hard",
                            "intent_label": "memory_management",
                            "estimated_training_value": "high",
                            "source": "synthetic",
                        },
                    ),
                    QAPair(
                        question="How to handle file not found errors?",
                        answer="Use try-except with proper error handling:\n```python\ntry:\n    with open('file.txt') as f:\n        data = f.read()\nexcept FileNotFoundError:\n    print('File not found')\n```",
                        metadata={
                            "id": "bug_003",
                            "difficulty": "medium",
                            "intent_label": "file_errors",
                            "estimated_training_value": "medium",
                            "source": "synthetic",
                        },
                    ),
                ],
                intents=["null_handling", "memory_management", "file_errors"],
            ),
            Dataset(
                type="testcase_generation",
                items=[
                    QAPair(
                        question="Write tests for a login function",
                        answer="```python\ndef test_login_success():\n    result = login('user', 'password')\n    assert result.status == 'success'\n\ndef test_login_failure():\n    result = login('user', 'wrong')\n    assert result.status == 'failure'\n```",
                        metadata={
                            "id": "test_001",
                            "difficulty": "medium",
                            "intent_label": "auth_testing",
                            "estimated_training_value": "high",
                            "source": "synthetic",
                        },
                    ),
                    QAPair(
                        question="How to test edge cases in sorting?",
                        answer="Test empty lists, single elements, already sorted, reverse sorted, and duplicates.",
                        metadata={
                            "id": "test_002",
                            "difficulty": "hard",
                            "intent_label": "edge_cases",
                            "estimated_training_value": "medium",
                            "source": "synthetic",
                        },
                    ),
                ],
                intents=["auth_testing", "edge_cases"],
            ),
        ],
        generation_method="llm",
        llm_provider="mock",
    )


class TestExportQAJsonl:
    """Tests for QA JSONL export."""

    def test_exports_all_items(self, sample_dataset_output):
        """Should export all items from all datasets."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            count = export_qa_jsonl(sample_dataset_output, output_path)
            assert count == 5  # 3 + 2 items
            
            # Verify file exists and has correct line count
            with open(output_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 5
        finally:
            output_path.unlink(missing_ok=True)

    def test_correct_format(self, sample_dataset_output):
        """Should export in correct QA format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_qa_jsonl(sample_dataset_output, output_path)
            
            with open(output_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            
            record = json.loads(first_line)
            assert "question" in record
            assert "answer" in record
            assert "metadata" in record
            assert record["question"] == "How do I fix a null pointer exception?"
        finally:
            output_path.unlink(missing_ok=True)

    def test_includes_dataset_type_in_metadata(self, sample_dataset_output):
        """Should include dataset_type in metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_qa_jsonl(sample_dataset_output, output_path)
            
            with open(output_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            
            record = json.loads(first_line)
            assert record["metadata"]["dataset_type"] == "bugfixing"
        finally:
            output_path.unlink(missing_ok=True)

    def test_valid_jsonl(self, sample_dataset_output):
        """Every line should be valid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_qa_jsonl(sample_dataset_output, output_path)
            
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)  # Should not raise
                    assert isinstance(record, dict)
        finally:
            output_path.unlink(missing_ok=True)


class TestExportInstructJsonl:
    """Tests for instruction JSONL export."""

    def test_exports_all_items(self, sample_dataset_output):
        """Should export all items."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            count = export_instruct_jsonl(sample_dataset_output, output_path)
            assert count == 5
        finally:
            output_path.unlink(missing_ok=True)

    def test_correct_format(self, sample_dataset_output):
        """Should export in instruction format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_instruct_jsonl(sample_dataset_output, output_path)
            
            with open(output_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            
            record = json.loads(first_line)
            assert "instruction" in record
            assert "input" in record
            assert "output" in record
            assert record["instruction"] == "How do I fix a null pointer exception?"
            assert record["input"] == ""
            assert "Check for null" in record["output"]
        finally:
            output_path.unlink(missing_ok=True)

    def test_includes_system_prompt(self, sample_dataset_output):
        """Should include system prompt in instruction when provided."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_instruct_jsonl(
                sample_dataset_output,
                output_path,
                system_prompt="You are a helpful coding assistant.",
            )
            
            with open(output_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            
            record = json.loads(first_line)
            assert "You are a helpful coding assistant." in record["instruction"]
        finally:
            output_path.unlink(missing_ok=True)


class TestExportChatJsonl:
    """Tests for chat JSONL export."""

    def test_exports_all_items(self, sample_dataset_output):
        """Should export all items."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            count = export_chat_jsonl(sample_dataset_output, output_path)
            assert count == 5
        finally:
            output_path.unlink(missing_ok=True)

    def test_correct_format(self, sample_dataset_output):
        """Should export in chat format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_chat_jsonl(sample_dataset_output, output_path)
            
            with open(output_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            
            record = json.loads(first_line)
            assert "messages" in record
            assert len(record["messages"]) == 3
            
            messages = record["messages"]
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"
            
            assert messages[1]["content"] == "How do I fix a null pointer exception?"
            assert "Check for null" in messages[2]["content"]
        finally:
            output_path.unlink(missing_ok=True)

    def test_custom_system_prompt(self, sample_dataset_output):
        """Should use custom system prompt."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_chat_jsonl(
                sample_dataset_output,
                output_path,
                system_prompt="You are a Python expert.",
            )
            
            with open(output_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            
            record = json.loads(first_line)
            assert record["messages"][0]["content"] == "You are a Python expert."
        finally:
            output_path.unlink(missing_ok=True)


class TestGoldenSetSampling:
    """Tests for golden set sampling."""

    def test_samples_correct_count(self, sample_dataset_output):
        """Should sample the correct number of items."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            counts = sample_golden_set(
                sample_dataset_output,
                output_path,
                top_unique=3,
                top_hard=2,
                top_code_heavy=2,
            )
            
            # Total should be <= top_unique + top_hard + top_code_heavy
            # (may be less due to deduplication)
            assert counts["total"] <= 7
            assert counts["total"] >= 1  # Should have at least some items
        finally:
            output_path.unlink(missing_ok=True)

    def test_avoids_duplicates(self, sample_dataset_output):
        """Should not include the same item in multiple categories."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            sample_golden_set(sample_dataset_output, output_path)
            
            with open(output_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Parse all items
            items = [json.loads(line) for line in lines]
            questions = [item["question"] for item in items]
            
            # No duplicate questions
            assert len(questions) == len(set(questions))
        finally:
            output_path.unlink(missing_ok=True)

    def test_includes_golden_category(self, sample_dataset_output):
        """Should include golden_category in metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            sample_golden_set(sample_dataset_output, output_path)
            
            with open(output_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            
            if first_line:
                record = json.loads(first_line)
                assert "golden_category" in record["metadata"]
                assert record["metadata"]["golden_category"] in [
                    "most_unique", "hardest", "code_heavy"
                ]
        finally:
            output_path.unlink(missing_ok=True)

    def test_empty_dataset(self):
        """Should handle empty datasets gracefully."""
        empty_output = DatasetOutput(
            project_summary="Empty test",
            datasets=[],
            generation_method="test",
            llm_provider="test",
        )
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            counts = sample_golden_set(empty_output, output_path)
            assert counts["total"] == 0
            
            # File should exist but be empty
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content == ""
        finally:
            output_path.unlink(missing_ok=True)


class TestExportAllFormats:
    """Tests for exporting all formats at once."""

    def test_creates_all_files(self, sample_dataset_output):
        """Should create all export files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            counts = export_all_formats(sample_dataset_output, output_dir)
            
            # Check all files exist
            assert (output_dir / "dataset_qa.jsonl").exists()
            assert (output_dir / "dataset_instruct.jsonl").exists()
            assert (output_dir / "dataset_chat.jsonl").exists()
            assert (output_dir / "golden_set.jsonl").exists()
            
            # Check counts
            assert counts["qa"] == 5
            assert counts["instruct"] == 5
            assert counts["chat"] == 5
            assert "golden_set" in counts

    def test_returns_correct_counts(self, sample_dataset_output):
        """Should return correct counts for each format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            counts = export_all_formats(sample_dataset_output, output_dir)
            
            assert counts["qa"] == 5
            assert counts["instruct"] == 5
            assert counts["chat"] == 5
            assert "golden_details" in counts
            assert "unique" in counts["golden_details"]
            assert "hard" in counts["golden_details"]
            assert "code_heavy" in counts["golden_details"]

    def test_creates_output_dir_if_not_exists(self, sample_dataset_output):
        """Should create output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nested" / "output"
            
            assert not output_dir.exists()
            
            export_all_formats(sample_dataset_output, output_dir)
            
            assert output_dir.exists()
            assert (output_dir / "dataset_qa.jsonl").exists()


class TestJsonlValidity:
    """Tests to ensure all exports produce valid JSONL."""

    def test_all_formats_produce_valid_jsonl(self, sample_dataset_output):
        """All export formats should produce valid JSONL files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            export_all_formats(sample_dataset_output, output_dir)
            
            for filename in [
                "dataset_qa.jsonl",
                "dataset_instruct.jsonl",
                "dataset_chat.jsonl",
                "golden_set.jsonl",
            ]:
                filepath = output_dir / filename
                with open(filepath, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                json.loads(line)
                            except json.JSONDecodeError as e:
                                pytest.fail(
                                    f"Invalid JSON in {filename} line {line_num}: {e}"
                                )

    def test_utf8_encoding(self):
        """Should properly handle UTF-8 characters."""
        # Create dataset with unicode content
        unicode_output = DatasetOutput(
            project_summary="Unicode test",
            datasets=[
                Dataset(
                    type="unicode_test",
                    items=[
                        QAPair(
                            question="How to handle émojis 🎉 in Python?",
                            answer="Use UTF-8 encoding: café, naïve, 日本語",
                            metadata={"id": "unicode_test", "difficulty": "easy"},
                        )
                    ],
                    intents=[],
                )
            ],
            generation_method="test",
            llm_provider="test",
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            export_all_formats(unicode_output, output_dir)
            
            # Read and parse to verify encoding
            with open(output_dir / "dataset_qa.jsonl", "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            record = json.loads(lines[0])
            assert "émojis" in record["question"]
            assert "🎉" in record["question"]
            assert "café" in record["answer"]
            assert "日本語" in record["answer"]
