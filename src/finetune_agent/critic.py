"""Self-critique agent for dataset quality filtering.

The Critic reviews generated datasets and identifies:
- Near-duplicate questions
- Trivial or low-signal answers
- Instruction-following violations
- Quality issues that would harm fine-tuning

Returns reject indices and improvement notes for optional regeneration.
"""

import re
from typing import Callable

from finetune_agent.schemas import CritiqueResult, Dataset, DatasetConstraints, DatasetOutput, QAPair


class DatasetCritic:
    """Self-critique agent for dataset quality assessment.
    
    Reviews generated Q&A pairs and identifies issues that would
    reduce the effectiveness of fine-tuning.
    """
    
    # Default thresholds for quality checks (can be overridden by constraints)
    DEFAULT_MIN_ANSWER_LENGTH = 50  # Minimum acceptable answer length
    DEFAULT_MIN_QUESTION_LENGTH = 10  # Minimum acceptable question length
    DEFAULT_SIMILARITY_THRESHOLD = 0.7  # Jaccard similarity threshold for duplicates
    
    def __init__(
        self,
        llm_client=None,
        aggressive: bool = False,
        progress_callback: Callable[[str], None] | None = None,
        constraints: DatasetConstraints | None = None,
    ):
        """Initialize the critic.
        
        Args:
            llm_client: Optional LLM client for enhanced critique
            aggressive: If True, apply stricter filtering criteria
            progress_callback: Optional callback for progress updates
            constraints: Optional dataset constraints for quality thresholds
        """
        self._llm = llm_client
        self._aggressive = aggressive
        self._progress_callback = progress_callback or (lambda x: None)
        self._constraints = constraints or DatasetConstraints()
        
        # Apply constraints to thresholds
        self.MIN_ANSWER_LENGTH = self._constraints.min_answer_length
        self.MIN_QUESTION_LENGTH = self.DEFAULT_MIN_QUESTION_LENGTH
        self.SIMILARITY_THRESHOLD = self._constraints.similarity_threshold
        self.BANNED_PHRASES = self._constraints.banned_phrases
        self.REQUIRE_CODE_RATIO = self._constraints.require_code_ratio
    
    @property
    def llm(self):
        """Lazy-load LLM client."""
        if self._llm is None:
            from finetune_agent.llm import get_llm_client
            self._llm = get_llm_client()
        return self._llm
    
    def _report_progress(self, message: str):
        """Report progress to callback."""
        self._progress_callback(message)
    
    def _get_ngrams(self, text: str, n: int = 3) -> set[tuple[str, ...]]:
        """Extract n-grams from text for similarity comparison."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < n:
            return {tuple(words)}
        return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _find_duplicates(self, items: list[QAPair]) -> list[tuple[int, int]]:
        """Find near-duplicate question pairs.
        
        Returns:
            List of (index1, index2) tuples for duplicate pairs
        """
        duplicates = []
        ngrams_cache = {}
        
        for i, item1 in enumerate(items):
            if i not in ngrams_cache:
                ngrams_cache[i] = self._get_ngrams(item1.question)
            
            for j in range(i + 1, len(items)):
                if j not in ngrams_cache:
                    ngrams_cache[j] = self._get_ngrams(items[j].question)
                
                similarity = self._jaccard_similarity(
                    ngrams_cache[i], 
                    ngrams_cache[j]
                )
                
                if similarity > self.SIMILARITY_THRESHOLD:
                    duplicates.append((i, j))
        
        return duplicates
    
    def _has_code(self, text: str) -> bool:
        """Check if text contains code snippets."""
        return (
            '```' in text or 
            'def ' in text or 
            'function ' in text.lower() or
            'class ' in text or
            'import ' in text or
            '=>' in text or  # Arrow functions
            '()' in text  # Function calls
        )
    
    def _extract_code_blocks(self, text: str) -> list[str]:
        """Extract all code blocks from text.
        
        Handles various formats:
        - ```python ... ```
        - ```py ... ```
        - ``` ... ``` (no language tag)
        - Code blocks with language tags on same line as content
        - Code blocks with newlines before/after backticks
        """
        code_blocks = []
        
        # Pattern 1: Standard fenced code blocks with optional language tag
        # Handles ```python, ```py, ```Python, or just ```
        pattern1 = r'```(?:python|py)?\s*\n(.*?)```'
        matches1 = re.findall(pattern1, text, re.DOTALL | re.IGNORECASE)
        code_blocks.extend(matches1)
        
        # Pattern 2: Code blocks with any language tag (captures content after tag)
        # This catches ```javascript, ```java, etc. but we still extract the code
        pattern2 = r'```\w*\s*\n(.*?)```'
        matches2 = re.findall(pattern2, text, re.DOTALL)
        code_blocks.extend(matches2)
        
        # Pattern 3: Code blocks with content on same line as opening backticks
        # e.g., ```python import pytest
        pattern3 = r'```(?:python|py)?\s*([^\n].*?)```'
        matches3 = re.findall(pattern3, text, re.DOTALL | re.IGNORECASE)
        code_blocks.extend(matches3)
        
        # Deduplicate while preserving order (some patterns may overlap)
        seen = set()
        unique_blocks = []
        for block in code_blocks:
            block_stripped = block.strip()
            if block_stripped and block_stripped not in seen:
                seen.add(block_stripped)
                unique_blocks.append(block_stripped)
        
        return unique_blocks
    
    def _check_pytest_contract(self, item: QAPair) -> list[str]:
        """Check if testcase_generation answer meets pytest contract.
        
        Contract requirements (all checked WITHIN code blocks):
        - At least one function with `def test_`
        - At least two `assert` statements
        - At least one of: pytest.mark.parametrize, pytest.raises, or fixture
        - Code inside a Python code block
        
        Returns:
            List of contract violation issues (empty if all requirements met)
        """
        issues = []
        answer = item.answer
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(answer)
        
        # Must have at least one code block
        if not code_blocks:
            issues.append("Missing code block (must contain ```python ... ```)")
            return issues  # Can't check other requirements without code
        
        # Combine all code for analysis
        all_code = '\n'.join(code_blocks)
        
        # Count test functions WITHIN code blocks
        test_func_count = len(re.findall(r'\bdef test_\w+', all_code))
        if test_func_count < 1:
            issues.append("Missing test function (must have at least one 'def test_')")
        
        # Count assert statements WITHIN code blocks
        assert_count = len(re.findall(r'\bassert\s+', all_code))
        if assert_count < 2:
            issues.append(f"Insufficient assertions (found {assert_count}, need at least 2)")
        
        # Check for pytest features WITHIN code blocks (at least one required)
        has_parametrize = bool(re.search(r'@?pytest\.mark\.parametrize', all_code))
        has_raises = 'pytest.raises' in all_code
        has_fixture_decorator = '@pytest.fixture' in all_code
        
        # Check for common fixture parameters in test function signatures
        common_fixtures = [
            'mocker', 'tmp_path', 'capsys', 'monkeypatch', 
            'request', 'capfd', 'caplog', 'recwarn', 'fixture'
        ]
        has_fixture_param = any(f in all_code for f in common_fixtures)
        
        # Check for test function with parameters (excluding empty parens and just self)
        fixture_pattern = r'def test_\w+\(\s*(?:self\s*,\s*)?([a-z_]\w*)'
        fixture_matches = re.findall(fixture_pattern, all_code)
        has_fixture_in_signature = any(
            m and m != 'self' for m in fixture_matches
        )
        
        has_fixture = has_fixture_decorator or has_fixture_param or has_fixture_in_signature
        
        if not (has_parametrize or has_raises or has_fixture):
            issues.append("Missing pytest feature (need pytest.mark.parametrize, pytest.raises, or fixture usage)")
        
        return issues
    
    def _check_answer_quality(self, item: QAPair, dataset_type: str = "") -> list[str]:
        """Check answer quality and return issues found."""
        issues = []
        answer = item.answer
        
        # Check minimum length
        if len(answer) < self.MIN_ANSWER_LENGTH:
            issues.append(f"Answer too short ({len(answer)} < {self.MIN_ANSWER_LENGTH} chars)")
        
        # Check for placeholder content
        placeholder_patterns = [
            r'\[.*?\]',  # [placeholder]
            r'TODO',
            r'FIXME',
            r'\.\.\.',  # Ellipsis as placeholder
            r'<.*?>',  # <placeholder>
        ]
        for pattern in placeholder_patterns:
            if re.search(pattern, answer):
                issues.append("Answer contains placeholder content")
                break
        
        # Check for empty or meaningless content
        stripped = re.sub(r'\s+', '', answer.lower())
        if len(stripped) < 20:
            issues.append("Answer lacks substantive content")
        
        # Check for banned phrases
        for phrase in self.BANNED_PHRASES:
            if phrase.lower() in answer.lower() or phrase.lower() in item.question.lower():
                issues.append(f"Contains banned phrase: '{phrase}'")
        
        # TESTCASE_GENERATION CONTRACT ENFORCEMENT
        if dataset_type in ("testcase_generation", "testcase"):
            pytest_issues = self._check_pytest_contract(item)
            issues.extend(pytest_issues)
        
        # For aggressive mode, check additional criteria
        if self._aggressive:
            # Check for code in technical answers
            if any(kw in item.question.lower() for kw in ['code', 'implement', 'fix', 'write']):
                if not self._has_code(answer):
                    issues.append("Technical answer lacks code examples")
            
            # Check for step-by-step in complex answers
            if 'how' in item.question.lower() and len(item.question) > 50:
                if not re.search(r'\d\.|step|first|then|finally', answer.lower()):
                    issues.append("Complex answer lacks structured steps")
        
        return issues
    
    def _check_question_quality(self, item: QAPair) -> list[str]:
        """Check question quality and return issues found."""
        issues = []
        question = item.question
        
        # Check minimum length
        if len(question) < self.MIN_QUESTION_LENGTH:
            issues.append("Question too short")
        
        # Check for vague questions
        vague_patterns = [
            r'^how\s*$',
            r'^what\s*$',
            r'^why\s*$',
            r'^explain\s*$',
        ]
        for pattern in vague_patterns:
            if re.match(pattern, question.lower().strip()):
                issues.append("Question too vague")
                break
        
        return issues
    
    def _llm_critique(self, items: list[QAPair]) -> CritiqueResult:
        """Use LLM for enhanced critique.
        
        Asks the LLM to identify quality issues that rule-based
        checks might miss.
        """
        self._report_progress("Running LLM-assisted critique...")
        
        # Format items for LLM
        items_text = []
        for i, item in enumerate(items[:20]):  # Limit to avoid token overload
            items_text.append(f"[{i}] Q: {item.question[:200]}\n    A: {item.answer[:300]}")
        
        prompt = f"""You are a dataset quality reviewer for fine-tuning datasets.

Review these Q&A pairs and identify issues:

{chr(10).join(items_text)}

Look for:
1. Near-duplicate questions (similar phrasing or intent)
2. Trivial answers that don't teach anything useful
3. Answers that don't properly address the question
4. Low training value items

Return JSON:
{{
  "reject_indices": [indices of items to reject],
  "improvement_notes": ["specific improvement suggestions"],
  "quality_assessment": "good|acceptable|needs_improvement",
  "duplicate_pairs": [[i, j], ...],
  "low_quality_indices": [indices of low quality items]
}}

Be constructive but thorough. Only reject items with clear issues."""

        try:
            response = self.llm.generate_json(prompt)
            return CritiqueResult(
                reject_indices=response.get("reject_indices", []),
                improvement_notes=response.get("improvement_notes", []),
                quality_assessment=response.get("quality_assessment", "acceptable"),
                duplicate_pairs=[
                    tuple(pair) for pair in response.get("duplicate_pairs", [])
                ],
                low_quality_indices=response.get("low_quality_indices", []),
            )
        except Exception as e:
            self._report_progress(f"LLM critique failed: {e}")
            return CritiqueResult()
    
    def critique_dataset(self, dataset: Dataset) -> CritiqueResult:
        """Critique a single dataset.
        
        Args:
            dataset: The dataset to critique
            
        Returns:
            CritiqueResult with reject indices and notes
        """
        self._report_progress(f"Critiquing dataset: {dataset.type}")
        
        reject_indices: set[int] = set()
        improvement_notes: list[str] = []
        low_quality_indices: list[int] = []
        
        # Rule-based checks
        # Track rejection reasons for reporting
        rejection_reasons: dict[str, int] = {
            "duplicate": 0,
            "too_short": 0,
            "missing_pytest_code": 0,
            "banned_phrases": 0,
            "low_quality": 0,
        }
        
        for i, item in enumerate(dataset.items):
            question_issues = self._check_question_quality(item)
            answer_issues = self._check_answer_quality(item, dataset_type=dataset.type)
            
            if question_issues or answer_issues:
                low_quality_indices.append(i)
                
                # Track specific rejection reasons
                # Pytest contract violation patterns in issue messages
                pytest_contract_patterns = [
                    "pytest", "test_", "assertion", "code block", "test function"
                ]
                
                for issue in answer_issues:
                    issue_lower = issue.lower()
                    if "too short" in issue_lower and "assertion" not in issue_lower:
                        rejection_reasons["too_short"] += 1
                    elif any(p in issue_lower for p in pytest_contract_patterns):
                        rejection_reasons["missing_pytest_code"] += 1
                    elif "banned phrase" in issue_lower:
                        rejection_reasons["banned_phrases"] += 1
                    else:
                        rejection_reasons["low_quality"] += 1
                
                # For testcase_generation, always reject items that fail pytest contract
                is_testcase = dataset.type in ("testcase_generation", "testcase")
                has_pytest_issues = any(
                    any(p in issue.lower() for p in pytest_contract_patterns)
                    for issue in answer_issues
                )
                
                # In aggressive mode OR if testcase_generation has pytest contract violations
                if self._aggressive or (is_testcase and has_pytest_issues):
                    reject_indices.add(i)
                    for issue in question_issues + answer_issues:
                        note = f"Item {i}: {issue}"
                        if note not in improvement_notes:
                            improvement_notes.append(note)
        
        # Find duplicates
        duplicate_pairs = self._find_duplicates(dataset.items)
        
        # For duplicates, keep the first, reject the second
        for i, j in duplicate_pairs:
            if self._aggressive:
                reject_indices.add(j)  # Reject the later duplicate
            improvement_notes.append(f"Items {i} and {j} are near-duplicates (similarity > {self.SIMILARITY_THRESHOLD})")
        
        # Check code ratio requirement
        if self.REQUIRE_CODE_RATIO > 0 and dataset.items:
            items_with_code = sum(1 for item in dataset.items if self._has_code(item.answer))
            actual_ratio = (items_with_code / len(dataset.items)) * 100
            
            if actual_ratio < self.REQUIRE_CODE_RATIO:
                improvement_notes.append(
                    f"Code ratio too low: {actual_ratio:.1f}% (required: {self.REQUIRE_CODE_RATIO}%)"
                )
                # In aggressive mode, reject items without code to boost ratio
                if self._aggressive:
                    for i, item in enumerate(dataset.items):
                        if not self._has_code(item.answer):
                            reject_indices.add(i)
                            if f"Item {i}: Missing code (code ratio enforcement)" not in improvement_notes:
                                improvement_notes.append(f"Item {i}: Missing code (code ratio enforcement)")
        
        # LLM-enhanced critique
        llm_result = self._llm_critique(dataset.items)
        
        # Merge LLM results
        if llm_result.reject_indices:
            if self._aggressive:
                reject_indices.update(llm_result.reject_indices)
            improvement_notes.extend(llm_result.improvement_notes)
        
        # Combine duplicate findings
        all_duplicates = list(set(duplicate_pairs + [
            tuple(p) for p in llm_result.duplicate_pairs
        ]))
        
        # Determine quality assessment
        reject_ratio = len(reject_indices) / len(dataset.items) if dataset.items else 0
        if reject_ratio > 0.3:
            quality = "needs_improvement"
        elif reject_ratio > 0.1:
            quality = "acceptable"
        else:
            quality = "good"
        
        return CritiqueResult(
            reject_indices=sorted(reject_indices),
            improvement_notes=improvement_notes,
            quality_assessment=quality,
            duplicate_pairs=all_duplicates,
            low_quality_indices=low_quality_indices,
        )
    
    def critique(self, output: DatasetOutput) -> dict[str, CritiqueResult]:
        """Critique all datasets in the output.
        
        Args:
            output: The complete dataset output
            
        Returns:
            Dictionary mapping dataset type to CritiqueResult
        """
        results = {}
        
        for dataset in output.datasets:
            results[dataset.type] = self.critique_dataset(dataset)
        
        return results
    
    def filter_dataset(
        self, 
        dataset: Dataset, 
        critique: CritiqueResult,
    ) -> Dataset:
        """Remove rejected items from a dataset.
        
        Args:
            dataset: The original dataset
            critique: The critique result
            
        Returns:
            New dataset with rejected items removed
        """
        if not critique.reject_indices:
            return dataset
        
        reject_set = set(critique.reject_indices)
        filtered_items = [
            item for i, item in enumerate(dataset.items)
            if i not in reject_set
        ]
        
        return Dataset(
            type=dataset.type,
            items=filtered_items,
            intents=dataset.intents,
        )
    
    def filter_all(
        self,
        output: DatasetOutput,
        critiques: dict[str, CritiqueResult],
    ) -> DatasetOutput:
        """Filter all datasets based on critique results.
        
        Args:
            output: The original dataset output
            critiques: Dictionary of critique results per dataset type
            
        Returns:
            New DatasetOutput with rejected items removed
        """
        filtered_datasets = []
        
        for dataset in output.datasets:
            critique = critiques.get(dataset.type, CritiqueResult())
            filtered = self.filter_dataset(dataset, critique)
            filtered_datasets.append(filtered)
        
        total_items = sum(len(d.items) for d in filtered_datasets)
        original_items = sum(len(d.items) for d in output.datasets)
        removed = original_items - total_items
        
        summary = output.project_summary
        if removed > 0:
            summary += f" After critique: {removed} items filtered out."
        
        return DatasetOutput(
            project_summary=summary,
            datasets=filtered_datasets,
            generation_method=output.generation_method,
            llm_provider=output.llm_provider,
        )
