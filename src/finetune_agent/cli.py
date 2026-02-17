"""Command-line interface for the finetune agent.

V2: Enhanced with model family selection, filtering options, and better progress.
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table

from finetune_agent.agent import FinetuneAgent
from finetune_agent.schemas import DatasetConstraints, ModelFamily, UserConstraints, UserProfile


# Force UTF-8 output on Windows
console = Console(force_terminal=True, legacy_windows=False)


def print_banner():
    """Print the application banner."""
    banner = """
+---------------------------------------------------------------+
|                  FINETUNE AGENT v2.0                          |
|         Agentic AI for Finetuning Engineering                 |
|                                                               |
|   This tool does NOT train models.                            |
|   It accelerates finetuning engineering.                      |
+---------------------------------------------------------------+
"""
    console.print(banner, style="bold cyan")


def get_user_input(profile: UserProfile | None = None) -> tuple[str, list[str], int, UserConstraints]:
    """Gather input from the user interactively.
    
    Args:
        profile: Optional saved user profile for defaults
        
    Returns:
        Tuple of (prompt, dataset_types, qa_per_type, constraints)
    """
    console.print()
    
    # Get defaults from profile
    default_types = "bugfixing,testcase_generation,doc_generation"
    default_count = "10"
    default_difficulty = "medium"
    default_tone = "technical"
    
    if profile:
        if profile.default_dataset_types:
            default_types = ",".join(profile.default_dataset_types)
        default_count = str(profile.default_qa_count)
        default_difficulty = profile.preferred_difficulty
        default_tone = profile.preferred_tone
    
    # Step 1: Main prompt
    console.print("[bold]Step 1: Describe your fine-tuning goal[/bold]", style="yellow")
    prompt = Prompt.ask(
        "\n[cyan]What do you want to fine-tune and what's the target behavior?[/cyan]\n",
        default="A code assistant that helps with Python debugging and error fixing",
    )
    
    console.print()
    
    # Step 2: Model family (V2)
    console.print("[bold]Step 2: Select target model family[/bold]", style="yellow")
    console.print("This helps optimize the dataset format and content.")
    
    model_choices = {
        "1": ModelFamily.CODE_LLM,
        "2": ModelFamily.CHAT_LLM,
        "3": ModelFamily.INSTRUCT,
        "4": ModelFamily.CLASSIFIER,
        "5": ModelFamily.OTHER,
    }
    
    console.print("  1. Code LLM (code generation, completion)")
    console.print("  2. Chat LLM (conversational, dialogue)")
    console.print("  3. Instruct (instruction-following)")
    console.print("  4. Classifier (categorization)")
    console.print("  5. Other")
    
    model_choice = Prompt.ask(
        "\n[cyan]Select model family[/cyan]",
        choices=["1", "2", "3", "4", "5"],
        default="1",
    )
    model_family = model_choices[model_choice]
    
    console.print()
    
    # Step 3: Dataset types
    console.print("[bold]Step 3: Choose dataset types[/bold]", style="yellow")
    console.print("Available: bugfixing, testcase_generation, doc_generation, code_review, refactoring")
    types_input = Prompt.ask(
        "\n[cyan]Enter dataset types (comma-separated)[/cyan]",
        default=default_types,
    )
    dataset_types = [t.strip() for t in types_input.split(",") if t.strip()]
    
    console.print()
    
    # Step 4: QA count
    console.print("[bold]Step 4: Set dataset size[/bold]", style="yellow")
    qa_count_input = Prompt.ask(
        "\n[cyan]How many Q&A pairs per dataset type?[/cyan]",
        default=default_count,
    )
    try:
        qa_per_type = int(qa_count_input)
        qa_per_type = max(1, min(100, qa_per_type))
    except ValueError:
        qa_per_type = 10
        console.print("[yellow]Invalid number, using default: 10[/yellow]")
    
    console.print()
    
    # Step 5: Constraints
    console.print("[bold]Step 5: Set generation constraints[/bold]", style="yellow")
    
    tone = Prompt.ask(
        "\n[cyan]Preferred tone[/cyan]",
        default=default_tone,
        choices=["technical", "casual", "formal"],
        show_choices=True,
    )
    
    difficulty = Prompt.ask(
        "[cyan]Difficulty level[/cyan]",
        default=default_difficulty,
        choices=["easy", "medium", "hard"],
        show_choices=True,
    )
    
    domain = Prompt.ask(
        "[cyan]Domain focus (optional, e.g., 'web development', 'data science')[/cyan]",
        default="",
    )
    
    console.print()
    
    # Step 6: Filtering (V2)
    console.print("[bold]Step 6: Quality filtering[/bold]", style="yellow")
    aggressive_filtering = Confirm.ask(
        "[cyan]Enable aggressive filtering? (stricter quality, may reduce output)[/cyan]",
        default=False,
    )
    
    additional = Prompt.ask(
        "[cyan]Any additional notes or constraints?[/cyan]",
        default="",
    )
    
    console.print()
    
    # Step 7: Advanced constraints (optional)
    console.print("[bold]Step 7: Advanced constraints (optional)[/bold]", style="yellow")
    use_advanced = Confirm.ask(
        "[cyan]Configure advanced quality constraints?[/cyan]",
        default=False,
    )
    
    dataset_constraints = DatasetConstraints()
    
    if use_advanced:
        console.print()
        console.print("[dim]Leave blank to use defaults.[/dim]")
        
        # Minimum answer length
        min_len_str = Prompt.ask(
            "[cyan]Minimum answer length (chars)[/cyan]",
            default="50",
        )
        try:
            dataset_constraints.min_answer_length = max(0, int(min_len_str))
        except ValueError:
            pass
        
        # Similarity threshold
        sim_str = Prompt.ask(
            "[cyan]Similarity threshold for duplicates (0.0-1.0)[/cyan]",
            default="0.7",
        )
        try:
            dataset_constraints.similarity_threshold = min(1.0, max(0.0, float(sim_str)))
        except ValueError:
            pass
        
        # Code ratio (for code LLMs)
        if model_family == ModelFamily.CODE_LLM:
            code_ratio_str = Prompt.ask(
                "[cyan]Minimum code ratio (0-100%)[/cyan]",
                default="0",
            )
            try:
                dataset_constraints.require_code_ratio = min(100, max(0, int(code_ratio_str)))
            except ValueError:
                pass
        
        # Banned phrases
        banned_str = Prompt.ask(
            "[cyan]Banned phrases (comma-separated, or blank)[/cyan]",
            default="",
        )
        if banned_str.strip():
            dataset_constraints.banned_phrases = [p.strip() for p in banned_str.split(",") if p.strip()]
        
        # Difficulty distribution
        set_dist = Confirm.ask(
            "[cyan]Set custom difficulty distribution?[/cyan]",
            default=False,
        )
        if set_dist:
            easy_str = Prompt.ask("[cyan]  Easy %[/cyan]", default="30")
            med_str = Prompt.ask("[cyan]  Medium %[/cyan]", default="50")
            hard_str = Prompt.ask("[cyan]  Hard %[/cyan]", default="20")
            try:
                easy = int(easy_str)
                med = int(med_str)
                hard = int(hard_str)
                if easy + med + hard == 100:
                    dataset_constraints.difficulty_distribution = {
                        "easy": easy, "medium": med, "hard": hard
                    }
                else:
                    console.print("[yellow]Distribution must sum to 100, using defaults.[/yellow]")
            except ValueError:
                console.print("[yellow]Invalid values, using defaults.[/yellow]")
    
    constraints = UserConstraints(
        tone=tone,
        difficulty=difficulty,
        domain=domain if domain else "",
        additional_notes=additional if additional else "",
        model_family=model_family,
        aggressive_filtering=aggressive_filtering,
        dataset_constraints=dataset_constraints,
    )
    
    return prompt, dataset_types, qa_per_type, constraints


def display_results(
    action_plan: str,
    evaluation,
    critiques: dict,
    output_path: Path,
):
    """Display the results of the agent run."""
    console.print()
    console.print("=" * 60, style="green")
    console.print("[bold green]Generation Complete![/bold green]")
    console.print("=" * 60, style="green")
    console.print()
    
    # Show action plan preview
    console.print("[bold]Action Plan Preview[/bold]", style="cyan")
    plan_lines = action_plan.split("\n")[:25]
    console.print(Markdown("\n".join(plan_lines)))
    if len(action_plan.split("\n")) > 25:
        console.print("[dim]... (see full plan in output folder)[/dim]")
    
    console.print()
    
    # Show critique summary
    console.print("[bold]Critique Summary[/bold]", style="cyan")
    critique_table = Table(show_header=True, header_style="bold")
    critique_table.add_column("Dataset", style="cyan")
    critique_table.add_column("Quality", justify="center")
    critique_table.add_column("Rejected", justify="right")
    critique_table.add_column("Duplicates", justify="right")
    
    for dtype, critique in critiques.items():
        quality = critique.quality_assessment
        quality_color = "green" if quality == "good" else "yellow" if quality == "acceptable" else "red"
        critique_table.add_row(
            dtype,
            f"[{quality_color}]{quality}[/{quality_color}]",
            str(len(critique.reject_indices)),
            str(len(critique.duplicate_pairs)),
        )
    
    console.print(critique_table)
    console.print()
    
    # Show evaluation results
    console.print("[bold]Evaluation Results[/bold]", style="cyan")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset Type", style="cyan")
    table.add_column("Items", justify="right")
    table.add_column("Uniqueness", justify="right")
    table.add_column("Lexical", justify="right")
    table.add_column("Structural", justify="right")
    table.add_column("Conceptual", justify="right")
    
    for eval_result in evaluation.dataset_evaluations:
        uniqueness_color = "green" if eval_result.uniqueness_score >= 70 else "yellow" if eval_result.uniqueness_score >= 50 else "red"
        table.add_row(
            eval_result.dataset_type,
            str(eval_result.item_count),
            f"[{uniqueness_color}]{eval_result.uniqueness_score:.1f}[/{uniqueness_color}]",
            f"{eval_result.lexical_score:.1f}",
            f"{eval_result.structural_score:.1f}",
            f"{eval_result.conceptual_score:.1f}",
        )
    
    console.print(table)
    console.print()
    
    # Health metrics
    if evaluation.health_metrics:
        console.print("[bold]Health Metrics[/bold]", style="cyan")
        hm = evaluation.health_metrics
        
        health_table = Table(show_header=False)
        health_table.add_column("Metric", style="dim")
        health_table.add_column("Value")
        
        health_table.add_row("Avg Answer Length", f"{hm.avg_answer_length:.0f} chars")
        health_table.add_row("Items with Code", f"{hm.items_with_code} ({hm.items_with_code_pct:.1f}%)")
        health_table.add_row("Intent Coverage", f"{hm.intent_coverage_score:.1f}%")
        
        if hm.difficulty_distribution:
            diff_str = ", ".join(f"{k}: {v}" for k, v in hm.difficulty_distribution.items())
            health_table.add_row("Difficulty Dist.", diff_str)
        
        console.print(health_table)
        console.print()
    
    # Overall rating
    rating_color = "green" if evaluation.overall_rating >= 70 else "yellow" if evaluation.overall_rating >= 50 else "red"
    console.print(Panel(
        f"[{rating_color} bold]Overall Rating: {evaluation.overall_rating:.1f}/100[/{rating_color} bold]",
        title="Quality Score",
        border_style=rating_color,
    ))
    
    console.print()
    
    # LLM Feedback (V2)
    if evaluation.llm_feedback:
        console.print("[bold]LLM Feedback[/bold]", style="cyan")
        console.print(f"  {evaluation.llm_feedback}")
        console.print()
    
    # Warnings
    if evaluation.warnings:
        console.print("[bold]Warnings[/bold]", style="red")
        for warning in evaluation.warnings:
            console.print(f"  [red]![/red] {warning}")
        console.print()
    
    # Feedback
    console.print("[bold]Recommendations[/bold]", style="cyan")
    for line in evaluation.feedback:
        if line.strip():
            console.print(f"  {line}")
    
    console.print()
    
    # Output location
    console.print(Panel(
        f"Output saved to: [bold]{output_path}[/bold]\n\n"
        "Files:\n"
        "  - action_plan.md - Detailed engineering plan\n"
        "  - dataset.json - Generated Q&A pairs with metadata\n"
        "  - evaluation.json - Quality scores and metrics\n"
        "  - critique.json - Self-critique results\n"
        "  - dataset_qa.jsonl - Simple Q&A format for training\n"
        "  - dataset_instruct.jsonl - Alpaca-style instruction format\n"
        "  - dataset_chat.jsonl - OpenAI chat format\n"
        "  - golden_set.jsonl - Curated top items for evaluation",
        title="Output Files",
        border_style="blue",
    ))


def show_recent_runs(agent: FinetuneAgent):
    """Show recent run history."""
    runs = agent.get_recent_runs(5)
    
    if not runs:
        console.print("[dim]No previous runs found.[/dim]")
        return
    
    console.print("\n[bold]Recent Runs[/bold]", style="cyan")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Run ID", style="cyan")
    table.add_column("Date", style="dim")
    table.add_column("Types")
    table.add_column("Rating", justify="right")
    table.add_column("Output")
    
    for run in runs:
        rating_color = "green" if run.overall_rating >= 70 else "yellow" if run.overall_rating >= 50 else "red"
        table.add_row(
            run.run_id,
            run.timestamp.strftime("%Y-%m-%d %H:%M"),
            ", ".join(run.dataset_types[:2]) + ("..." if len(run.dataset_types) > 2 else ""),
            f"[{rating_color}]{run.overall_rating:.1f}[/{rating_color}]",
            str(Path(run.output_path).name),
        )
    
    console.print(table)
    console.print()


class ProgressTracker:
    """Track progress with Rich progress bar."""
    
    def __init__(self, console: Console):
        self.console = console
        self.current_message = ""
        self.progress = None
        self.task = None
    
    def __call__(self, message: str):
        """Update progress message."""
        self.current_message = message
        if self.task is not None and self.progress is not None:
            self.progress.update(self.task, description=message)


def main():
    """Main entry point for the CLI."""
    print_banner()
    
    # Create progress tracker
    progress_tracker = ProgressTracker(console)
    
    # Initialize agent with progress callback
    agent = FinetuneAgent(progress_callback=progress_tracker)
    
    # Check for profile
    profile = agent.get_user_profile()
    if profile:
        console.print(f"[dim]Welcome back! Using saved preferences.[/dim]")
    
    # Show LLM provider info
    try:
        from finetune_agent.llm import get_llm_client
        llm = get_llm_client()
        console.print(f"[dim]LLM Provider: {llm.provider_name}[/dim]")
    except Exception:
        console.print("[dim]LLM Provider: mock (no API key configured)[/dim]")
    
    # Show recent runs
    show_recent_runs(agent)
    
    try:
        # Get user input
        prompt, dataset_types, qa_per_type, constraints = get_user_input(profile)
        
        console.print()
        console.print("=" * 60, style="blue")
        console.print("[bold blue]Starting Generation Pipeline...[/bold blue]")
        console.print("=" * 60, style="blue")
        console.print()
        
        # Run the agent with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress_tracker.progress = progress
            progress_tracker.task = progress.add_task("Initializing...", total=None)
            
            action_plan, dataset, evaluation, critiques, output_path = agent.run_interactive(
                prompt=prompt,
                dataset_types=dataset_types,
                qa_per_type=qa_per_type,
                constraints=constraints,
            )
            
            progress.update(progress_tracker.task, description="[green]Complete!")
        
        # Display results
        display_results(action_plan, evaluation, critiques, output_path)
        
        # Ask to save preferences
        save_prefs = Prompt.ask(
            "\n[cyan]Save your preferences for next time?[/cyan]",
            choices=["y", "n"],
            default="y",
        )
        
        if save_prefs.lower() == "y":
            new_profile = UserProfile(
                default_dataset_types=dataset_types,
                preferred_difficulty=constraints.difficulty,
                preferred_tone=constraints.tone,
                default_qa_count=qa_per_type,
            )
            agent.save_user_profile(new_profile)
            console.print("[green]Preferences saved![/green]")
        
        console.print("\n[bold green]Done! Happy fine-tuning![/bold green]\n")
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
