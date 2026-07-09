"""Agent module - orchestrates the Planner → Generator → Critic → Evaluator pipeline.

V2: Now includes self-critique and optional regeneration of rejected items.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from distillery.critic import DatasetCritic
from distillery.dataset_generator import DatasetGenerator
from distillery.evaluator import Evaluator
from distillery.memory import get_memory_store
from distillery.planner import Planner
from distillery.schemas import (
    CritiqueResult,
    DatasetOutput,
    GenerationRequest,
    OverallEvaluation,
    RunSummary,
    UserConstraints,
    UserProfile,
)
from distillery.exporter import export_all_formats
from distillery.utils import (
    create_output_dir,
    generate_run_id,
    save_json,
    save_markdown,
    truncate_text,
)


class GenerationError(Exception):
    """Raised when dataset generation fails critically."""
    pass


class FinetuneAgent:
    """Main agent class that orchestrates the fine-tuning workflow.
    
    V2 Pipeline:
    1. Planner: Generates an action plan based on user requirements
    2. Generator: Creates Q&A datasets (LLM-backed or template-based)
    3. Critic: Reviews and filters low-quality items
    4. Refill Loop: Regenerate to meet requested count (max 3 iterations)
    5. Evaluator: Scores the final datasets for quality
    
    Ensures requested item count is always met when possible.
    """
    
    MAX_REFILL_ITERATIONS = 3  # Maximum regeneration attempts to meet count
    
    def __init__(
        self,
        seed: int | None = None,
        llm_client=None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """Initialize the agent.
        
        Args:
            seed: Random seed for reproducibility
            llm_client: Optional LLM client (uses default if None)
            progress_callback: Optional callback for progress updates
        """
        self._seed = seed
        self._llm = llm_client
        self._progress_callback = progress_callback or (lambda x: None)
        
        # Initialize components (lazy-loaded with LLM)
        self._planner = None
        self._generator = None
        self._critic = None
        self._evaluator = None
        
        self.memory = get_memory_store()
    
    def _report_progress(self, message: str):
        """Report progress to callback."""
        self._progress_callback(message)
    
    @property
    def llm(self):
        """Lazy-load LLM client."""
        if self._llm is None:
            from distillery.llm import get_llm_client
            self._llm = get_llm_client()
        return self._llm
    
    @property
    def planner(self) -> Planner:
        """Get or create the planner."""
        if self._planner is None:
            self._planner = Planner(llm_client=self.llm)
        return self._planner
    
    @property
    def generator(self) -> DatasetGenerator:
        """Get or create the generator."""
        if self._generator is None:
            self._generator = DatasetGenerator(
                llm_client=self.llm,
                seed=self._seed,
                progress_callback=self._progress_callback,
            )
        return self._generator
    
    @property
    def critic(self) -> DatasetCritic:
        """Get or create the critic."""
        if self._critic is None:
            self._critic = DatasetCritic(
                llm_client=self.llm,
                aggressive=False,  # Default; overridden per-run
                progress_callback=self._progress_callback,
            )
        return self._critic
    
    @property
    def evaluator(self) -> Evaluator:
        """Get or create the evaluator."""
        if self._evaluator is None:
            self._evaluator = Evaluator(
                llm_client=self.llm,
                progress_callback=self._progress_callback,
            )
        return self._evaluator
    
    def get_user_profile(self) -> UserProfile | None:
        """Get the stored user profile."""
        return self.memory.get_profile()
    
    def save_user_profile(self, profile: UserProfile) -> None:
        """Save user profile preferences."""
        self.memory.save_profile(profile)
    
    def get_recent_runs(self, limit: int = 5) -> list[RunSummary]:
        """Get recent run summaries."""
        return self.memory.get_recent_runs(limit)
    
    def _track_rejections(
        self,
        dataset: DatasetOutput,
        critiques: dict[str, CritiqueResult],
        debug_info: dict[str, Any],
    ) -> None:
        """Track rejection statistics and capture sample rejections.
        
        For each rejection reason, captures one representative sample with
        question and answer_snippet (truncated to 400 chars).
        
        Args:
            dataset: The dataset output containing all items
            critiques: Critique results per dataset type
            debug_info: Debug info dict to update in place
        """
        # Build a map of dataset type -> items
        datasets_by_type = {ds.type: ds.items for ds in dataset.datasets}
        
        for dtype, critique in critiques.items():
            debug_info["rejected_count"][dtype] = len(critique.reject_indices)
            
            items = datasets_by_type.get(dtype, [])
            if not items:
                continue
            
            # Track reasons and sample rejections
            reasons: dict[str, int] = {}
            samples: dict[str, dict[str, str]] = {}
            
            for idx in critique.reject_indices:
                if idx >= len(items):
                    continue
                    
                item = items[idx]

                # Correctness-gate verdicts have no structural "issue" strings,
                # so report them directly from the critic's stored verdicts.
                code_verdicts = getattr(self.critic, "_code_verdicts", {})
                code_verdict = code_verdicts.get(dtype, {}).get(idx)
                if code_verdict is not None and code_verdict.is_reject:
                    reason_key = code_verdict.verdict.value
                    reasons[reason_key] = reasons.get(reason_key, 0) + 1
                    if reason_key not in samples:
                        samples[reason_key] = {
                            "question": item.question[:400],
                            "answer_snippet": item.answer[:400],
                        }

                # Re-check to get specific rejection reasons
                answer_issues = self.critic._check_answer_quality(item, dataset_type=dtype)
                question_issues = self.critic._check_question_quality(item)
                all_issues = answer_issues + question_issues
                
                # Categorize reasons
                for issue in all_issues:
                    issue_lower = issue.lower()
                    reason_key = None
                    
                    if "code block" in issue_lower:
                        reason_key = "missing_code_block"
                    elif "test function" in issue_lower or "def test_" in issue_lower:
                        reason_key = "missing_test_function"
                    elif "assertion" in issue_lower:
                        reason_key = "insufficient_assertions"
                    elif "pytest feature" in issue_lower:
                        reason_key = "missing_pytest_feature"
                    elif "duplicate" in issue_lower:
                        reason_key = "duplicate"
                    elif "too short" in issue_lower:
                        reason_key = "too_short"
                    elif "banned phrase" in issue_lower:
                        reason_key = "banned_phrase"
                    else:
                        reason_key = "other"
                    
                    reasons[reason_key] = reasons.get(reason_key, 0) + 1
                    
                    # Capture one sample per reason
                    if reason_key not in samples:
                        samples[reason_key] = {
                            "question": item.question[:400],
                            "answer_snippet": item.answer[:400],
                        }
            
            debug_info["top_rejection_reasons"][dtype] = reasons
            debug_info["sample_rejections"][dtype] = samples

    def _summarize_verification(self, debug_info: dict[str, Any]) -> None:
        """Roll up the correctness gate's per-item verdicts into a coverage summary.

        Turns the gate into a headline stat: of the code items graded, how many were
        actually executed, how many passed, how many were rejected as broken, and how
        many were skipped (couldn't be run because of external deps). Only populated
        when the gate ran (validate_generated_code enabled).
        """
        from distillery.critic_execution import Verdict

        code_verdicts = getattr(self.critic, "_code_verdicts", {})
        for dtype, verdicts in code_verdicts.items():
            if not verdicts:
                continue
            values = list(verdicts.values())
            passed = sum(1 for v in values if v.verdict == Verdict.OK)
            exec_fail = sum(1 for v in values if v.verdict == Verdict.EXEC_FAIL)
            rejected = sum(1 for v in values if v.is_reject)
            skipped = sum(1 for v in values if v.verdict == Verdict.SKIPPED_EXTERNAL_DEPS)
            debug_info["verification_coverage"][dtype] = {
                "graded": len(values),
                "executed": passed + exec_fail,  # actually run under pytest
                "passed": passed,
                "rejected": rejected,
                "skipped_external_deps": skipped,
            }

    def run(
        self,
        prompt: str,
        dataset_types: list[str],
        qa_per_type: int,
        constraints: UserConstraints | None = None,
        output_dir: Path | None = None,
        use_llm: bool = True,
        regenerate_rejected: bool = False,
    ) -> tuple[str, DatasetOutput, OverallEvaluation, dict[str, CritiqueResult], Path, dict[str, Any]]:
        """Run the complete fine-tuning agent pipeline.
        
        V2 Pipeline:
        1. Planning → Generate action plan
        2. Generation → Create Q&A datasets
        3. Critique → Review and identify issues
        4. Refill Loop → Regenerate to meet requested count
        5. Filtering → Remove rejected items
        6. Evaluation → Score final dataset
        
        Args:
            prompt: User description of what they want to fine-tune
            dataset_types: List of dataset type labels
            qa_per_type: Number of Q&A pairs to generate per type
            constraints: Optional constraints for generation
            output_dir: Optional custom output directory
            use_llm: Whether to use LLM for generation (default True)
            regenerate_rejected: Whether to regenerate rejected items (default False)
            
        Returns:
            Tuple of (action_plan, dataset, evaluation, critiques, output_path, debug_info)
        """
        if constraints is None:
            constraints = UserConstraints()
        
        # Ensure qa_per_type is valid
        if qa_per_type <= 0:
            raise ValueError(f"qa_per_type must be > 0, got {qa_per_type}")
        
        # Initialize debug info for troubleshooting
        debug_info: dict[str, Any] = {
            "requested_count_per_type": qa_per_type,
            "dataset_types": dataset_types,
            "generated_count_before_critique": {},
            "rejected_count": {},
            "accepted_count": {},
            "refill_iterations_run": 0,
            "final_count": {},
            "top_rejection_reasons": {},
            "sample_rejections": {},  # { dtype: { reason: { question, answer_snippet } } }
            "first_item_answer_snippet": {},  # { dtype: snippet }
            "verification_coverage": {},  # { dtype: {graded, executed, passed, rejected, skipped_external_deps} }
            "errors": [],
        }
        
        # Create generation request
        request = GenerationRequest(
            prompt=prompt,
            dataset_types=dataset_types,
            qa_per_type=qa_per_type,
            constraints=constraints,
            use_llm=use_llm,
        )
        
        # =====================================================================
        # Honesty check: surface the resolved LLM provider/model BEFORE any
        # generation so a silent fallback to mock can never go unnoticed.
        # =====================================================================
        provider = getattr(self.llm, "provider_name", "unknown")
        model = (
            getattr(self.llm, "model_name", None)
            or getattr(self.llm, "_model", None)
            or "n/a"
        )
        debug_info["llm_provider"] = provider
        debug_info["llm_model"] = model
        print(f"Generation provider: {provider} | model: {model}")
        if provider == "mock":
            print(
                "WARNING: LLM provider resolved to 'mock' -- generated content is "
                "canned/templated, NOT from a real model. Set LLM_PROVIDER=openai "
                "(with OPENAI_API_KEY) or LLM_PROVIDER=ollama to use a real model."
            )

        # =====================================================================
        # Phase 1: Planning
        # =====================================================================
        self._report_progress("Phase 1: Generating action plan...")
        action_plan = self.planner.generate_action_plan(request)
        
        # =====================================================================
        # Phase 2: Generation
        # =====================================================================
        self._report_progress("Phase 2: Generating datasets...")
        dataset = self.generator.generate(request)
        
        # Track generated counts BEFORE critique
        for ds in dataset.datasets:
            debug_info["generated_count_before_critique"][ds.type] = len(ds.items)
            # Capture first item answer snippet for debugging
            if ds.items:
                first_answer = ds.items[0].answer
                debug_info["first_item_answer_snippet"][ds.type] = first_answer[:400]
        
        # Early failure detection: check if generation produced any items
        total_generated = sum(len(ds.items) for ds in dataset.datasets)
        if total_generated == 0:
            error_msg = "Generation produced 0 items. Check LLM provider and prompts."
            debug_info["errors"].append(error_msg)
            self._report_progress(f"ERROR: {error_msg}")
        
        # =====================================================================
        # Phase 3: Critique and Refill Loop
        # =====================================================================
        self._report_progress("Phase 3: Running self-critique...")
        
        # Update critic with current constraints (including new DatasetConstraints)
        self._critic = DatasetCritic(
            llm_client=self.llm,
            aggressive=constraints.aggressive_filtering,
            progress_callback=self._progress_callback,
            constraints=constraints.dataset_constraints,
            execute_tests=constraints.validate_generated_code,
        )
        
        critiques = self.critic.critique(dataset)
        
        # Track rejection info and capture sample rejections
        self._track_rejections(dataset, critiques, debug_info)
        
        # =====================================================================
        # Phase 4: Refill Loop - Ensure requested count is met
        # =====================================================================
        count_warnings: list[str] = []
        iteration = 0
        
        while iteration < self.MAX_REFILL_ITERATIONS:
            # Filter to get current valid items
            filtered_dataset = self.critic.filter_all(dataset, critiques)
            
            # Check if any dataset is short
            needs_refill = False
            for ds in filtered_dataset.datasets:
                current_count = len(ds.items)
                if current_count < qa_per_type:
                    needs_refill = True
                    shortfall = qa_per_type - current_count
                    self._report_progress(
                        f"Phase 4: Refilling {ds.type} (iteration {iteration + 1}/{self.MAX_REFILL_ITERATIONS}): "
                        f"need {shortfall} more items to reach {qa_per_type}"
                    )
            
            if not needs_refill:
                break
            
            # Regenerate to fill gaps
            dataset = self._refill_to_count(request, filtered_dataset, qa_per_type)
            
            # Re-critique and track rejections
            critiques = self.critic.critique(dataset)
            self._track_rejections(dataset, critiques, debug_info)
            iteration += 1
        
        debug_info["refill_iterations_run"] = iteration
        
        # Final filtering
        self._report_progress("Phase 5: Final filtering...")
        filtered_dataset = self.critic.filter_all(dataset, critiques)
        
        # Track final counts and accepted counts
        for ds in filtered_dataset.datasets:
            debug_info["final_count"][ds.type] = len(ds.items)
            debug_info["accepted_count"][ds.type] = len(ds.items)

        # Roll up the correctness gate's per-item verdicts into a coverage summary
        # (only populated when validate_generated_code was enabled).
        self._summarize_verification(debug_info)
        
        # Check final counts and add warnings
        for ds in filtered_dataset.datasets:
            current_count = len(ds.items)
            if current_count < qa_per_type:
                warning = (
                    f"Dataset '{ds.type}' has {current_count} items (requested {qa_per_type}) "
                    f"after {self.MAX_REFILL_ITERATIONS} regeneration attempts"
                )
                count_warnings.append(warning)
                self._report_progress(f"Warning: {warning}")
            
            # Critical warning for 0 items
            if current_count == 0:
                error_msg = f"CRITICAL: Dataset '{ds.type}' has 0 items after all processing"
                debug_info["errors"].append(error_msg)
                self._report_progress(f"ERROR: {error_msg}")

        # Honesty check: fail loudly if EVERYTHING was filtered out instead of
        # silently writing an empty dataset and continuing.
        total_final = sum(len(ds.items) for ds in filtered_dataset.datasets)
        if total_final == 0:
            total_generated = sum(debug_info["generated_count_before_critique"].values())
            total_rejected = sum(debug_info["rejected_count"].values())
            reason_parts = []
            for dtype, reasons in debug_info["top_rejection_reasons"].items():
                if reasons:
                    reason_str = ", ".join(f"{k}={v}" for k, v in reasons.items())
                    reason_parts.append(f"{dtype}: {reason_str}")
            reasons_summary = "; ".join(reason_parts) or "no rejection reasons recorded"
            raise GenerationError(
                f"Final dataset is empty: 0 items after critique and "
                f"{self.MAX_REFILL_ITERATIONS} refill attempt(s). "
                f"Generated {total_generated} item(s), rejected {total_rejected}. "
                f"Top rejection reasons -> {reasons_summary}. "
                f"(provider={provider}, model={model})"
            )

        # =====================================================================
        # Phase 6: Evaluation
        # =====================================================================
        self._report_progress("Phase 6: Evaluating dataset quality...")
        evaluation = self.evaluator.evaluate(filtered_dataset, requested_count=qa_per_type)
        
        # Add count warnings to evaluation
        if count_warnings:
            evaluation.warnings.extend(count_warnings)
        
        # =====================================================================
        # Save Outputs
        # =====================================================================
        self._report_progress("Saving outputs...")
        
        if output_dir is None:
            output_dir = create_output_dir()
        
        save_markdown(action_plan, output_dir, "action_plan.md")
        save_json(filtered_dataset.model_dump(mode="json"), output_dir, "dataset.json")
        save_json(evaluation.model_dump(mode="json"), output_dir, "evaluation.json")
        
        # Save critique results
        critique_output = {
            dtype: critique.model_dump(mode="json")
            for dtype, critique in critiques.items()
        }
        save_json(critique_output, output_dir, "critique.json")
        
        # Save debug info for troubleshooting
        save_json(debug_info, output_dir, "debug.json")
        
        # =====================================================================
        # Phase 7: Export to JSONL Formats
        # =====================================================================
        self._report_progress("Phase 7: Exporting to JSONL formats...")
        
        export_counts = export_all_formats(
            filtered_dataset,
            output_dir,
            system_prompt="You are a helpful assistant.",
        )
        
        self._report_progress(
            f"Exported: {export_counts['qa']} QA, "
            f"{export_counts['instruct']} Instruct, "
            f"{export_counts['chat']} Chat, "
            f"{export_counts['golden_set']} Golden Set items"
        )
        
        # Save run to memory
        run_id = generate_run_id()
        run_summary = RunSummary(
            run_id=run_id,
            timestamp=datetime.now(),
            prompt=truncate_text(prompt, 100),
            dataset_types=dataset_types,
            qa_per_type=qa_per_type,
            overall_rating=evaluation.overall_rating,
            output_path=str(output_dir),
        )
        self.memory.add_run(run_summary)
        
        self._report_progress("Complete!")
        
        return action_plan, filtered_dataset, evaluation, critiques, output_dir, debug_info
    
    def _refill_to_count(
        self,
        request: GenerationRequest,
        filtered_dataset: DatasetOutput,
        target_count: int,
    ) -> DatasetOutput:
        """Refill datasets to meet the target count.
        
        For each dataset that has fewer items than requested, generate
        additional items to fill the gap.
        
        Args:
            request: Original generation request
            filtered_dataset: Dataset after filtering (may have fewer items)
            target_count: Target number of items per dataset
            
        Returns:
            Updated dataset with additional items to meet count
        """
        updated_datasets = []
        
        for ds in filtered_dataset.datasets:
            current_count = len(ds.items)
            
            if current_count >= target_count:
                # Already have enough items
                updated_datasets.append(ds)
                continue
            
            # Calculate how many more we need
            shortfall = target_count - current_count
            self._report_progress(f"Generating {shortfall} additional items for {ds.type}...")
            
            # Generate replacement items
            regen_request = GenerationRequest(
                prompt=request.prompt,
                dataset_types=[ds.type],
                qa_per_type=shortfall,
                constraints=request.constraints,
                use_llm=request.use_llm,
            )
            
            try:
                regen_output = self.generator.generate(regen_request)
                if regen_output.datasets:
                    new_items = regen_output.datasets[0].items
                else:
                    new_items = []
            except Exception as e:
                self._report_progress(f"Refill generation failed: {e}")
                new_items = []
            
            # Combine existing items with new ones
            combined_items = list(ds.items) + list(new_items)
            
            from distillery.schemas import Dataset
            updated_datasets.append(Dataset(
                type=ds.type,
                items=combined_items,
                intents=ds.intents,
            ))
        
        return DatasetOutput(
            project_summary=filtered_dataset.project_summary + " (refilled)",
            datasets=updated_datasets,
            generation_method=filtered_dataset.generation_method,
            llm_provider=filtered_dataset.llm_provider,
        )
    
    def run_interactive(
        self,
        prompt: str,
        dataset_types: list[str],
        qa_per_type: int,
        constraints: UserConstraints | None = None,
    ) -> tuple[str, DatasetOutput, OverallEvaluation, dict[str, CritiqueResult], Path, dict[str, Any]]:
        """Run with progress tracking for interactive CLI.
        
        This is a wrapper around run() that uses default settings
        appropriate for interactive use.
        """
        return self.run(
            prompt=prompt,
            dataset_types=dataset_types,
            qa_per_type=qa_per_type,
            constraints=constraints,
            use_llm=True,
            regenerate_rejected=False,  # Faster for interactive use
        )
