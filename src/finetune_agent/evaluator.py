"""Evaluator module for scoring dataset quality.

V2: Now includes LLM-assisted conceptual scoring and health metrics.
"""

import math
import re
from collections import Counter
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from finetune_agent.schemas import (
    DatasetOutput,
    EvaluationResult,
    HealthMetrics,
    OverallEvaluation,
)


class Evaluator:
    """Evaluates the quality of generated datasets.
    
    V2 Scoring combines:
    - Lexical score (50%): TF-IDF diversity + n-gram uniqueness
    - Structural score (30%): Question variety + length distribution
    - Conceptual score (20%): LLM-assessed semantic diversity
    
    Final uniqueness_score = 0.5 * lexical + 0.3 * structural + 0.2 * conceptual
    """
    
    # V2 Weights for uniqueness components
    UNIQUENESS_WEIGHTS = {
        "lexical": 0.50,
        "structural": 0.30,
        "conceptual": 0.20,
    }
    
    # Weights for overall rating
    OVERALL_WEIGHTS = {
        "uniqueness": 0.40,
        "length_sanity": 0.25,
        "coverage": 0.35,
    }
    
    def __init__(
        self,
        llm_client=None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """Initialize the evaluator.
        
        Args:
            llm_client: Optional LLM client for conceptual scoring
            progress_callback: Optional callback for progress updates
        """
        self._llm = llm_client
        self._progress_callback = progress_callback or (lambda x: None)
        self._tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2),
        )
    
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
    
    # =========================================================================
    # Lexical Scoring (TF-IDF + N-grams)
    # =========================================================================
    
    def _calculate_lexical_diversity(self, texts: list[str]) -> float:
        """Calculate lexical diversity using type-token ratio."""
        if not texts:
            return 0.0
        
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        unique_words = set(all_words)
        ttr = len(unique_words) / len(all_words)
        length_factor = min(1.0, len(all_words) / 500)
        adjusted_ttr = ttr * (1 + length_factor * 0.3)
        
        return min(100.0, adjusted_ttr * 100)
    
    def _calculate_ngram_uniqueness(self, texts: list[str], n: int = 3) -> float:
        """Calculate uniqueness based on n-gram overlap."""
        if len(texts) < 2:
            return 100.0
        
        def get_ngrams(text: str, n: int) -> set[tuple]:
            words = re.findall(r'\b\w+\b', text.lower())
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
        
        ngram_sets = [get_ngrams(text, n) for text in texts]
        
        total_overlap = 0
        comparisons = 0
        
        for i in range(len(ngram_sets)):
            for j in range(i + 1, len(ngram_sets)):
                if ngram_sets[i] and ngram_sets[j]:
                    overlap = len(ngram_sets[i] & ngram_sets[j])
                    union = len(ngram_sets[i] | ngram_sets[j])
                    if union > 0:
                        total_overlap += overlap / union
                        comparisons += 1
        
        if comparisons == 0:
            return 100.0
        
        avg_overlap = total_overlap / comparisons
        return (1 - avg_overlap) * 100
    
    def _calculate_tfidf_diversity(self, texts: list[str]) -> float:
        """Calculate diversity using TF-IDF cosine similarity."""
        if len(texts) < 2:
            return 100.0
        
        try:
            tfidf_matrix = self._tfidf.fit_transform(texts)
            similarities = cosine_similarity(tfidf_matrix)
            
            total_sim = 0
            count = 0
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    total_sim += similarities[i, j]
                    count += 1
            
            if count == 0:
                return 100.0
            
            avg_similarity = total_sim / count
            return (1 - avg_similarity) * 100
        except Exception:
            return 50.0
    
    def calculate_lexical_score(self, texts: list[str]) -> float:
        """Calculate combined lexical score.
        
        Combines TF-IDF diversity, n-gram uniqueness, and vocabulary richness.
        """
        if not texts or len(texts) < 2:
            return 100.0 if texts else 0.0
        
        lexical_div = self._calculate_lexical_diversity(texts)
        ngram_uniq = self._calculate_ngram_uniqueness(texts)
        tfidf_div = self._calculate_tfidf_diversity(texts)
        
        # Weighted combination
        score = (0.3 * lexical_div + 0.35 * ngram_uniq + 0.35 * tfidf_div)
        return min(100.0, max(0.0, score))
    
    # =========================================================================
    # Structural Scoring
    # =========================================================================
    
    def _calculate_entropy(self, counts: list[int]) -> float:
        """Calculate Shannon entropy for a distribution."""
        total = sum(counts)
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counts:
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _calculate_structural_variety(self, texts: list[str]) -> float:
        """Calculate variety in text structure."""
        if not texts:
            return 0.0
        
        # Question word variety
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        question_word_counts = Counter()
        
        for text in texts:
            lower_text = text.lower()
            for qw in question_words:
                if lower_text.startswith(qw) or f" {qw} " in lower_text:
                    question_word_counts[qw] += 1
                    break
        
        if question_word_counts:
            qw_entropy = self._calculate_entropy(list(question_word_counts.values()))
            max_entropy = self._calculate_entropy([1] * len(question_words))
            qw_variety = (qw_entropy / max_entropy) * 100 if max_entropy > 0 else 0
        else:
            qw_variety = 50.0
        
        # Length distribution variety
        lengths = [len(text) for text in texts]
        if len(set(lengths)) == 1:
            length_variety = 0.0
        else:
            mean_length = sum(lengths) / len(lengths)
            variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
            std_dev = variance ** 0.5
            cv = std_dev / mean_length if mean_length > 0 else 0
            length_variety = min(100.0, cv * 200)
        
        return (qw_variety + length_variety) / 2
    
    def calculate_structural_score(self, texts: list[str]) -> float:
        """Calculate structural variety score."""
        return self._calculate_structural_variety(texts)
    
    # =========================================================================
    # Conceptual Scoring (LLM-assisted)
    # =========================================================================
    
    def calculate_conceptual_score(self, texts: list[str]) -> float:
        """Calculate conceptual diversity using LLM.
        
        Asks the LLM to assess semantic diversity that lexical
        measures might miss.
        """
        if not texts or len(texts) < 2:
            return 100.0 if texts else 0.0
        
        self._report_progress("Calculating conceptual diversity with LLM...")
        
        # Sample texts for LLM (to avoid token limits)
        sample_size = min(15, len(texts))
        sample = texts[:sample_size]
        
        prompt = f"""Analyze the conceptual diversity of these {len(sample)} text samples from a fine-tuning dataset.

Samples:
{chr(10).join(f'{i+1}. {t[:200]}...' if len(t) > 200 else f'{i+1}. {t}' for i, t in enumerate(sample))}

Rate the conceptual diversity from 0-100 based on:
- Do the samples cover different topics/concepts?
- Are there varied approaches to similar problems?
- Is there thematic richness beyond surface-level differences?

Return JSON:
{{
  "conceptual_diversity_score": <number 0-100>,
  "reasoning": "<brief explanation>",
  "strengths": ["<strength 1>", ...],
  "weaknesses": ["<weakness 1>", ...]
}}

Be objective and constructive."""

        try:
            response = self.llm.generate_json(prompt)
            score = response.get("conceptual_diversity_score", 70)
            return min(100.0, max(0.0, float(score)))
        except Exception as e:
            self._report_progress(f"LLM conceptual scoring failed: {e}")
            # Fall back to structural variety as proxy
            return self._calculate_structural_variety(texts)
    
    # =========================================================================
    # Health Metrics
    # =========================================================================
    
    def calculate_health_metrics(self, dataset: DatasetOutput) -> HealthMetrics:
        """Calculate dataset health metrics.
        
        Returns metrics about answer length, difficulty distribution,
        intent coverage, and code presence.
        """
        all_answer_lengths = []
        difficulty_counts: Counter = Counter()
        intent_counts: Counter = Counter()
        items_with_code = 0
        total_items = 0
        
        for ds in dataset.datasets:
            for item in ds.items:
                total_items += 1
                all_answer_lengths.append(len(item.answer))
                
                # Check for code
                if "```" in item.answer or "def " in item.answer:
                    items_with_code += 1
                
                # Extract metadata
                metadata = item.metadata
                difficulty = metadata.get("difficulty", "medium")
                intent = metadata.get("intent_label", "unknown")
                
                difficulty_counts[difficulty] += 1
                intent_counts[intent] += 1
        
        # Calculate intent coverage (entropy-based)
        if intent_counts:
            intent_entropy = self._calculate_entropy(list(intent_counts.values()))
            max_entropy = math.log2(len(intent_counts)) if len(intent_counts) > 1 else 1
            intent_coverage = (intent_entropy / max_entropy * 100) if max_entropy > 0 else 0
        else:
            intent_coverage = 0
        
        return HealthMetrics(
            avg_answer_length=sum(all_answer_lengths) / len(all_answer_lengths) if all_answer_lengths else 0,
            difficulty_distribution=dict(difficulty_counts),
            intent_coverage_score=round(intent_coverage, 2),
            items_with_code=items_with_code,
            items_with_code_pct=round(items_with_code / total_items * 100, 2) if total_items > 0 else 0,
        )
    
    # =========================================================================
    # Combined Uniqueness Score (V2)
    # =========================================================================
    
    def calculate_uniqueness_score(
        self, 
        texts: list[str],
        include_conceptual: bool = True,
    ) -> tuple[float, float, float, float]:
        """Calculate the overall uniqueness score with component breakdown.
        
        Args:
            texts: List of text strings to evaluate
            include_conceptual: Whether to include LLM conceptual score
            
        Returns:
            Tuple of (overall_score, lexical_score, structural_score, conceptual_score)
        """
        if not texts:
            return 0.0, 0.0, 0.0, 0.0
        
        if len(texts) == 1:
            return 100.0, 100.0, 100.0, 100.0
        
        lexical = self.calculate_lexical_score(texts)
        structural = self.calculate_structural_score(texts)
        
        if include_conceptual:
            conceptual = self.calculate_conceptual_score(texts)
        else:
            conceptual = structural  # Use structural as fallback
        
        # Weighted combination per V2 spec
        score = (
            self.UNIQUENESS_WEIGHTS["lexical"] * lexical +
            self.UNIQUENESS_WEIGHTS["structural"] * structural +
            self.UNIQUENESS_WEIGHTS["conceptual"] * conceptual
        )
        
        return (
            min(100.0, max(0.0, score)),
            round(lexical, 2),
            round(structural, 2),
            round(conceptual, 2),
        )
    
    # =========================================================================
    # Length and Coverage Scoring
    # =========================================================================
    
    def _calculate_length_sanity(self, dataset: DatasetOutput) -> float:
        """Check if lengths are reasonable."""
        scores = []
        
        for ds in dataset.datasets:
            for item in ds.items:
                q_len = len(item.question)
                if q_len < 10:
                    q_score = q_len / 10 * 50
                elif q_len > 500:
                    q_score = max(0, 100 - (q_len - 500) / 10)
                else:
                    q_score = 100
                
                a_len = len(item.answer)
                if a_len < 50:
                    a_score = a_len / 50 * 50
                elif a_len > 5000:
                    a_score = max(0, 100 - (a_len - 5000) / 100)
                else:
                    a_score = 100
                
                scores.append((q_score + a_score) / 2)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_coverage(self, dataset: DatasetOutput) -> float:
        """Calculate coverage across different aspects."""
        if not dataset.datasets:
            return 0.0
        
        type_counts = [len(ds.items) for ds in dataset.datasets]
        
        if not type_counts or sum(type_counts) == 0:
            return 0.0
        
        mean_count = sum(type_counts) / len(type_counts)
        variance = sum((c - mean_count) ** 2 for c in type_counts) / len(type_counts)
        cv = (variance ** 0.5) / mean_count if mean_count > 0 else 1
        
        distribution_score = max(0, 100 - cv * 100)
        min_count = min(type_counts)
        min_count_score = min(100, min_count * 10)
        
        return (distribution_score + min_count_score) / 2
    
    def _calculate_count_fulfillment(
        self, 
        dataset: DatasetOutput, 
        requested_count: int,
    ) -> float:
        """Calculate how well the dataset meets the requested count.
        
        Returns:
            Float between 0.0 and 1.0 representing fulfillment ratio
        """
        if requested_count <= 0 or not dataset.datasets:
            return 1.0
        
        total_requested = len(dataset.datasets) * requested_count
        total_actual = sum(len(ds.items) for ds in dataset.datasets)
        
        fulfillment = total_actual / total_requested if total_requested > 0 else 1.0
        return min(1.0, fulfillment)  # Cap at 1.0 (can't be better than 100%)
    
    # =========================================================================
    # Feedback Generation
    # =========================================================================
    
    def _generate_feedback(
        self,
        dataset: DatasetOutput,
        evaluations: list[EvaluationResult],
        overall_rating: float,
        health_metrics: HealthMetrics,
    ) -> list[str]:
        """Generate qualitative feedback about the dataset."""
        feedback = []
        
        # Overall assessment
        if overall_rating >= 80:
            feedback.append("✅ Overall quality is excellent.")
        elif overall_rating >= 60:
            feedback.append("⚠️ Overall quality is good but has room for improvement.")
        else:
            feedback.append("❌ Overall quality needs significant improvement.")
        
        # Uniqueness feedback per dataset
        for eval_result in evaluations:
            if eval_result.uniqueness_score < 50:
                feedback.append(f"❌ Low uniqueness in '{eval_result.dataset_type}'. Consider more variety.")
            elif eval_result.uniqueness_score < 70:
                feedback.append(f"⚠️ Moderate uniqueness in '{eval_result.dataset_type}'.")
        
        # Health metrics feedback
        if health_metrics.avg_answer_length < 100:
            feedback.append("⚠️ Average answer length is short. Consider more detailed answers.")
        
        if health_metrics.items_with_code_pct < 30:
            feedback.append("⚠️ Low percentage of items with code examples.")
        
        # Difficulty distribution feedback
        diff_dist = health_metrics.difficulty_distribution
        if diff_dist:
            total = sum(diff_dist.values())
            if diff_dist.get("easy", 0) / total > 0.6:
                feedback.append("⚠️ Most items are 'easy'. Consider more challenging examples.")
            if diff_dist.get("hard", 0) / total < 0.1:
                feedback.append("⚠️ Few 'hard' items. Consider adding advanced examples.")
        
        # Intent coverage feedback
        if health_metrics.intent_coverage_score < 50:
            feedback.append("⚠️ Low intent coverage. Intents are not evenly distributed.")
        
        # Size feedback
        total_items = sum(len(ds.items) for ds in dataset.datasets)
        if total_items < 20:
            feedback.append("⚠️ Dataset is small. Consider generating more examples.")
        
        # Recommendations
        feedback.append("\n📋 Recommendations:")
        feedback.append("  • Review Q&A pairs for domain accuracy")
        feedback.append("  • Ensure answers teach distinct concepts")
        feedback.append("  • Balance difficulty levels for progressive learning")
        
        return feedback
    
    def _generate_llm_feedback(self, dataset: DatasetOutput) -> str:
        """Generate LLM-based qualitative feedback."""
        self._report_progress("Generating LLM feedback...")
        
        # Sample items for feedback
        sample_items = []
        for ds in dataset.datasets:
            for item in ds.items[:3]:
                sample_items.append(f"Q: {item.question[:100]}\nA: {item.answer[:200]}...")
        
        if not sample_items:
            return ""
        
        prompt = f"""Review this fine-tuning dataset sample and provide brief, actionable feedback.

Sample items:
{chr(10).join(sample_items[:6])}

Provide 2-3 sentences of constructive feedback on:
1. Quality and usefulness for fine-tuning
2. Any patterns that could be improved
3. Specific suggestions for enhancement

Be concise and practical."""

        try:
            response = self.llm.generate(prompt, max_tokens=300)
            return response.strip()
        except Exception:
            return ""
    
    def _generate_warnings(self, dataset: DatasetOutput, health_metrics: HealthMetrics) -> list[str]:
        """Generate warnings about potential dataset issues."""
        warnings = []
        
        total_items = sum(len(ds.items) for ds in dataset.datasets)
        
        if total_items < 10:
            warnings.append("Very small dataset may lead to poor fine-tuning results")
        
        if health_metrics.avg_answer_length < 50:
            warnings.append("Answers are too short for effective training")
        
        # Check for empty datasets
        for ds in dataset.datasets:
            if len(ds.items) == 0:
                warnings.append(f"Dataset type '{ds.type}' is empty")
        
        # Check for potential data leakage indicators
        for ds in dataset.datasets:
            for item in ds.items:
                if "test data" in item.answer.lower() or "example answer" in item.answer.lower():
                    warnings.append("Some answers may contain placeholder or test content")
                    break
        
        return warnings
    
    # =========================================================================
    # Main Evaluation Method
    # =========================================================================
    
    def evaluate(
        self, 
        dataset: DatasetOutput, 
        requested_count: int | None = None,
    ) -> OverallEvaluation:
        """Evaluate the quality of a dataset.
        
        V2: Includes LLM-assisted scoring and health metrics.
        V2.1: Penalizes datasets that fail to meet requested count.
        
        Args:
            dataset: The generated dataset to evaluate
            requested_count: Optional requested items per dataset (for count penalty)
            
        Returns:
            OverallEvaluation containing scores, metrics, and feedback
        """
        self._report_progress("Evaluating dataset quality...")
        
        evaluations = []
        all_uniqueness_scores = []
        
        for ds in dataset.datasets:
            self._report_progress(f"Evaluating dataset type: {ds.type}")
            
            # Combine questions and answers for evaluation
            texts = [f"{item.question} {item.answer}" for item in ds.items]
            
            # Calculate uniqueness with component breakdown
            uniqueness, lexical, structural, conceptual = self.calculate_uniqueness_score(texts)
            all_uniqueness_scores.append(uniqueness)
            
            # Calculate lengths
            q_lengths = [len(item.question) for item in ds.items]
            a_lengths = [len(item.answer) for item in ds.items]
            
            # Calculate per-dataset health metrics
            ds_health = self._calculate_dataset_health(ds)
            
            evaluations.append(EvaluationResult(
                dataset_type=ds.type,
                uniqueness_score=round(uniqueness, 2),
                item_count=len(ds.items),
                avg_question_length=round(sum(q_lengths) / len(q_lengths), 2) if q_lengths else 0,
                avg_answer_length=round(sum(a_lengths) / len(a_lengths), 2) if a_lengths else 0,
                lexical_score=lexical,
                structural_score=structural,
                conceptual_score=conceptual,
                health_metrics=ds_health,
            ))
        
        # Calculate overall rating
        avg_uniqueness = sum(all_uniqueness_scores) / len(all_uniqueness_scores) if all_uniqueness_scores else 0
        length_sanity = self._calculate_length_sanity(dataset)
        coverage = self._calculate_coverage(dataset)
        
        overall_rating = (
            self.OVERALL_WEIGHTS["uniqueness"] * avg_uniqueness +
            self.OVERALL_WEIGHTS["length_sanity"] * length_sanity +
            self.OVERALL_WEIGHTS["coverage"] * coverage
        )
        
        # Apply count penalty if requested_count is specified
        if requested_count and requested_count > 0:
            count_fulfillment = self._calculate_count_fulfillment(dataset, requested_count)
            # Penalize by up to 20% for not meeting requested count
            count_penalty = (1 - count_fulfillment) * 20
            overall_rating = overall_rating - count_penalty
        
        overall_rating = min(100.0, max(0.0, overall_rating))
        
        # Calculate overall health metrics
        health_metrics = self.calculate_health_metrics(dataset)
        
        # Generate feedback
        feedback = self._generate_feedback(dataset, evaluations, overall_rating, health_metrics)
        
        # Generate LLM feedback
        llm_feedback = self._generate_llm_feedback(dataset)
        
        # Generate warnings
        warnings = self._generate_warnings(dataset, health_metrics)
        
        return OverallEvaluation(
            dataset_evaluations=evaluations,
            overall_rating=round(overall_rating, 2),
            feedback=feedback,
            health_metrics=health_metrics,
            llm_feedback=llm_feedback,
            warnings=warnings,
        )
    
    def _calculate_dataset_health(self, ds) -> HealthMetrics:
        """Calculate health metrics for a single dataset."""
        answer_lengths = [len(item.answer) for item in ds.items]
        difficulty_counts: Counter = Counter()
        intent_counts: Counter = Counter()
        items_with_code = 0
        
        for item in ds.items:
            if "```" in item.answer or "def " in item.answer:
                items_with_code += 1
            
            metadata = item.metadata
            difficulty_counts[metadata.get("difficulty", "medium")] += 1
            intent_counts[metadata.get("intent_label", "unknown")] += 1
        
        # Intent coverage
        if intent_counts and len(intent_counts) > 1:
            intent_entropy = self._calculate_entropy(list(intent_counts.values()))
            max_entropy = math.log2(len(intent_counts))
            intent_coverage = (intent_entropy / max_entropy * 100) if max_entropy > 0 else 0
        else:
            intent_coverage = 0
        
        total = len(ds.items)
        
        return HealthMetrics(
            avg_answer_length=sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
            difficulty_distribution=dict(difficulty_counts),
            intent_coverage_score=round(intent_coverage, 2),
            items_with_code=items_with_code,
            items_with_code_pct=round(items_with_code / total * 100, 2) if total > 0 else 0,
        )
