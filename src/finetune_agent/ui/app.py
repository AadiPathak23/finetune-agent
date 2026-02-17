"""Streamlit UI for Finetune Agent.

Run with: streamlit run src/finetune_agent/ui/app.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from finetune_agent.agent import FinetuneAgent
from finetune_agent.exporter import export_chat_jsonl, export_instruct_jsonl, export_qa_jsonl
from finetune_agent.schemas import DatasetConstraints, ModelFamily, UserConstraints


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Finetune Agent",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = None  # Will be created with selected provider
    if "results" not in st.session_state:
        st.session_state.results = None
    if "progress_messages" not in st.session_state:
        st.session_state.progress_messages = []
    if "running" not in st.session_state:
        st.session_state.running = False
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "mock"
    if "ollama_host" not in st.session_state:
        st.session_state.ollama_host = "http://localhost:11434"
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = "qwen2.5-coder"
    if "ollama_status" not in st.session_state:
        st.session_state.ollama_status = None


init_session_state()


# =============================================================================
# Helper Functions
# =============================================================================

def check_ollama_connection(host: str, model: str) -> tuple[bool, str]:
    """Check Ollama connection status."""
    try:
        from finetune_agent.llm.ollama import OllamaClient
        client = OllamaClient(host=host, model=model)
        return client.check_connection()
    except Exception as e:
        return False, f"Error: {e}"


def get_llm_client_for_provider(provider: str, ollama_host: str = None, ollama_model: str = None):
    """Get LLM client based on selected provider."""
    from finetune_agent.llm import get_llm_client
    
    if provider == "ollama":
        return get_llm_client(
            provider="ollama",
            host=ollama_host or st.session_state.ollama_host,
            model=ollama_model or st.session_state.ollama_model,
        )
    else:
        return get_llm_client(provider=provider)


# =============================================================================
# Header
# =============================================================================

st.title("🔧 Finetune Agent")
st.markdown("**Agentic AI for finetuning engineering**")
st.markdown("---")


# =============================================================================
# Layout: Two Columns
# =============================================================================

left_col, right_col = st.columns([1, 2])


# =============================================================================
# Left Column: Inputs
# =============================================================================

with left_col:
    # =========================================================================
    # LLM Provider Configuration
    # =========================================================================
    st.header("🤖 LLM Provider")
    
    provider_options = ["mock", "openai", "ollama"]
    provider_labels = {
        "mock": "Mock (Testing)",
        "openai": "OpenAI API",
        "ollama": "Ollama (Local)",
    }
    
    selected_provider = st.selectbox(
        "Provider",
        options=provider_options,
        index=provider_options.index(st.session_state.llm_provider),
        format_func=lambda x: provider_labels.get(x, x),
        help="Select the LLM provider for generation",
    )
    st.session_state.llm_provider = selected_provider
    
    # Provider-specific configuration
    if selected_provider == "ollama":
        with st.expander("🦙 Ollama Settings", expanded=True):
            ollama_host = st.text_input(
                "Host URL",
                value=st.session_state.ollama_host,
                help="Ollama server URL (default: http://localhost:11434)",
            )
            st.session_state.ollama_host = ollama_host
            
            ollama_model = st.text_input(
                "Model",
                value=st.session_state.ollama_model,
                help="Model name (e.g., qwen2.5-coder, llama3.2, mistral)",
            )
            st.session_state.ollama_model = ollama_model
            
            # Check connection button
            if st.button("🔄 Check Connection", use_container_width=True):
                with st.spinner("Checking Ollama connection..."):
                    is_connected, message = check_ollama_connection(ollama_host, ollama_model)
                    st.session_state.ollama_status = (is_connected, message)
            
            # Show status
            if st.session_state.ollama_status is not None:
                is_connected, message = st.session_state.ollama_status
                if is_connected:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
                    st.markdown("""
                    **Setup Steps:**
                    1. Install Ollama: [ollama.com/download](https://ollama.com/download)
                    2. Start server: `ollama serve`
                    3. Pull model: `ollama pull qwen2.5-coder`
                    """)
    
    elif selected_provider == "openai":
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            st.success("✅ OpenAI API key configured")
        else:
            st.warning("⚠️ Set OPENAI_API_KEY environment variable")
    
    elif selected_provider == "mock":
        st.info("ℹ️ Using mock LLM for testing (no API calls)")
    
    st.markdown("---")
    
    # =========================================================================
    # Dataset Configuration
    # =========================================================================
    st.header("📊 Configuration")
    
    # Main prompt
    prompt = st.text_area(
        "Describe what you want to train",
        value="A Python debugging assistant that helps fix common errors and exceptions",
        height=120,
        help="Describe the target behavior and use case for your fine-tuned model",
    )
    
    # Dataset types
    available_types = [
        "bugfixing",
        "testcase_generation",
        "doc_generation",
        "code_review",
        "refactoring",
    ]
    
    dataset_types = st.multiselect(
        "Dataset types",
        options=available_types,
        default=["bugfixing", "testcase_generation"],
        help="Select which types of Q&A pairs to generate",
    )
    
    # Q&A count
    qa_per_type = st.slider(
        "Q&A pairs per dataset type",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Number of question-answer pairs to generate for each dataset type",
    )
    
    # Model family
    model_family_options = {
        "Code LLM": ModelFamily.CODE_LLM,
        "Chat LLM": ModelFamily.CHAT_LLM,
        "Instruction-Following": ModelFamily.INSTRUCT,
        "Classifier": ModelFamily.CLASSIFIER,
        "Other": ModelFamily.OTHER,
    }
    
    model_family_choice = st.selectbox(
        "Target model family",
        options=list(model_family_options.keys()),
        index=0,
        help="The type of model you're fine-tuning",
    )
    model_family = model_family_options[model_family_choice]
    
    # Difficulty
    difficulty = st.selectbox(
        "Difficulty level",
        options=["easy", "medium", "hard"],
        index=1,
    )
    
    # Tone
    tone = st.selectbox(
        "Tone",
        options=["technical", "casual", "formal"],
        index=0,
    )
    
    # Aggressive filtering
    aggressive_filtering = st.toggle(
        "Aggressive filtering",
        value=False,
        help="Enable stricter quality thresholds (may reduce output quantity)",
    )
    
    # Domain (optional)
    domain = st.text_input(
        "Domain focus (optional)",
        value="",
        placeholder="e.g., web development, data science",
    )
    
    st.markdown("---")
    
    # ==========================================================================
    # Advanced Constraints Expander
    # ==========================================================================
    with st.expander("⚙️ Advanced Constraints", expanded=False):
        st.caption("Fine-tune quality control settings for production datasets.")
        
        # Minimum answer length
        min_answer_length = st.slider(
            "Minimum answer length (chars)",
            min_value=0,
            max_value=500,
            value=50,
            step=10,
            help="Answers shorter than this will be flagged by the critic",
        )
        
        # Similarity threshold
        similarity_threshold = st.slider(
            "Similarity threshold for duplicates",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher values = stricter duplicate detection (0.0-1.0)",
        )
        
        # Code ratio requirement
        require_code_ratio = st.slider(
            "Minimum code ratio (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Minimum percentage of answers that must contain code (for code LLMs)",
        )
        
        # Banned phrases
        banned_phrases_str = st.text_input(
            "Banned phrases (comma-separated)",
            value="",
            placeholder="e.g., TODO, FIXME, placeholder",
            help="Phrases that should be flagged by the critic",
        )
        banned_phrases = [p.strip() for p in banned_phrases_str.split(",") if p.strip()]
        
        # Difficulty distribution
        st.markdown("**Difficulty Distribution Target**")
        st.caption("Must sum to 100%")
        
        diff_col1, diff_col2, diff_col3 = st.columns(3)
        with diff_col1:
            easy_pct = st.number_input("Easy %", min_value=0, max_value=100, value=30, step=5)
        with diff_col2:
            medium_pct = st.number_input("Medium %", min_value=0, max_value=100, value=50, step=5)
        with diff_col3:
            hard_pct = st.number_input("Hard %", min_value=0, max_value=100, value=20, step=5)
        
        total_pct = easy_pct + medium_pct + hard_pct
        if total_pct != 100:
            st.warning(f"Distribution sums to {total_pct}% (should be 100%)")
        
        difficulty_distribution = {"easy": easy_pct, "medium": medium_pct, "hard": hard_pct}
    
    st.markdown("---")
    
    # Run button
    run_disabled = len(dataset_types) == 0 or len(prompt.strip()) == 0
    
    # Show current provider
    st.caption(f"Using: **{provider_labels.get(selected_provider, selected_provider)}**")
    
    if st.button(
        "🚀 Run Agent",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    ):
        st.session_state.running = True
        st.session_state.progress_messages = []
        
        # Progress callback
        def progress_callback(message: str):
            st.session_state.progress_messages.append(message)
        
        # Get LLM client based on selected provider
        try:
            llm_client = get_llm_client_for_provider(
                selected_provider,
                ollama_host=st.session_state.ollama_host if selected_provider == "ollama" else None,
                ollama_model=st.session_state.ollama_model if selected_provider == "ollama" else None,
            )
        except Exception as e:
            st.error(f"Failed to initialize LLM client: {e}")
            st.session_state.running = False
            st.stop()
        
        # Create agent with selected LLM client
        agent = FinetuneAgent(
            seed=42,
            llm_client=llm_client,
            progress_callback=progress_callback,
        )
        
        # Build dataset constraints
        dataset_constraints = DatasetConstraints(
            min_answer_length=min_answer_length,
            similarity_threshold=similarity_threshold,
            require_code_ratio=require_code_ratio,
            banned_phrases=banned_phrases,
            difficulty_distribution=difficulty_distribution,
        )
        
        # Build constraints
        constraints = UserConstraints(
            tone=tone,
            difficulty=difficulty,
            domain=domain,
            model_family=model_family,
            aggressive_filtering=aggressive_filtering,
            dataset_constraints=dataset_constraints,
        )
        
        # Run the agent
        with st.spinner("Running agent pipeline..."):
            try:
                action_plan, dataset, evaluation, critiques, output_path, debug_info = agent.run(
                    prompt=prompt,
                    dataset_types=dataset_types,
                    qa_per_type=qa_per_type,
                    constraints=constraints,
                    use_llm=True,
                )
                
                st.session_state.results = {
                    "action_plan": action_plan,
                    "dataset": dataset,
                    "evaluation": evaluation,
                    "critiques": critiques,
                    "output_path": output_path,
                    "debug_info": debug_info,
                    "requested_qa_per_type": qa_per_type,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                # Store agent for recent runs history
                st.session_state.agent = agent
                st.session_state.running = False
                st.rerun()
                
            except Exception as e:
                st.error(f"Error running agent: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state.running = False
    
    # Show recent runs from memory
    st.markdown("---")
    st.subheader("Recent Runs")
    
    if st.session_state.agent is not None:
        recent_runs = st.session_state.agent.get_recent_runs(3)
        if recent_runs:
            for run in recent_runs:
                with st.expander(f"Run {run.run_id} — {run.timestamp.strftime('%m/%d %H:%M')}"):
                    st.write(f"**Types**: {', '.join(run.dataset_types)}")
                    st.write(f"**Rating**: {run.overall_rating:.1f}/100")
                    st.write(f"**Output**: `{Path(run.output_path).name}`")
        else:
            st.caption("No previous runs found.")
    else:
        st.caption("No previous runs found.")


# =============================================================================
# Right Column: Outputs
# =============================================================================

with right_col:
    st.header("Results")
    
    if st.session_state.results is None:
        st.info("Configure your dataset and click **Run Agent** to generate results.")
        
        # Show example output structure
        with st.expander("What will be generated?"):
            st.markdown("""
            The agent will generate:
            
            1. **Action Plan** — A professional engineering document with:
               - Target model analysis
               - Dataset design rationale
               - Risk assessment
               - Implementation roadmap
            
            2. **Dataset** — Q&A pairs in JSON format with metadata:
               - Question and answer text
               - Difficulty level
               - Intent labels
               - Estimated training value
            
            3. **Critique** — Self-review identifying:
               - Near-duplicate questions
               - Low-quality items
               - Improvement suggestions
            
            4. **Evaluation** — Quality scores including:
               - Uniqueness score (lexical + structural + conceptual)
               - Overall rating
               - Health metrics
               - Warnings
            """)
    
    else:
        results = st.session_state.results
        debug_info = results.get("debug_info", {})
        requested_count = results.get("requested_qa_per_type", 0)
        
        # Summary bar
        eval_result = results["evaluation"]
        rating = eval_result.overall_rating
        rating_color = "green" if rating >= 70 else "orange" if rating >= 50 else "red"
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_items = sum(len(ds.items) for ds in results["dataset"].datasets)
            st.metric("Total Items", total_items)
        with col2:
            st.metric("Dataset Types", len(results["dataset"].datasets))
        with col3:
            st.metric("Overall Rating", f"{rating:.1f}/100")
        with col4:
            st.metric("Output", Path(results["output_path"]).name)
        
        # =====================================================================
        # ERROR PANEL: Show when total_items == 0
        # =====================================================================
        if total_items == 0:
            st.error("🚨 **CRITICAL: No items generated!** All items were rejected by the critic.")
            
            with st.expander("🔍 Debug Information", expanded=True):
                st.markdown("### Generation Summary")
                st.write(f"**Requested per type**: {debug_info.get('requested_count_per_type', 'N/A')}")
                
                st.markdown("**Generated before critique:**")
                for dtype, count in debug_info.get("generated_count_before_critique", {}).items():
                    st.write(f"- {dtype}: {count}")
                
                st.markdown("**Rejected counts:**")
                for dtype, count in debug_info.get("rejected_count", {}).items():
                    st.write(f"- {dtype}: {count}")
                
                st.markdown("**Top Rejection Reasons:**")
                for dtype, reasons in debug_info.get("top_rejection_reasons", {}).items():
                    st.write(f"**{dtype}:**")
                    for reason, count in reasons.items():
                        st.write(f"  - {reason}: {count}")
                
                st.write(f"**Refill iterations run**: {debug_info.get('refill_iterations_run', 0)}")
                
                # Sample rejections - show representative rejected items per reason
                sample_rejections = debug_info.get("sample_rejections", {})
                if sample_rejections:
                    st.markdown("### Sample Rejected Items")
                    st.caption("One representative rejected item per rejection reason:")
                    
                    for dtype, samples in sample_rejections.items():
                        st.markdown(f"**{dtype}:**")
                        for reason, sample in samples.items():
                            with st.expander(f"🔴 {reason}"):
                                st.markdown("**Question:**")
                                st.code(sample.get("question", "N/A"), language=None)
                                st.markdown("**Answer (snippet):**")
                                st.code(sample.get("answer_snippet", "N/A"), language=None)
                
                # First item answer snippet for debugging LLM output format
                first_snippets = debug_info.get("first_item_answer_snippet", {})
                if first_snippets:
                    st.markdown("### First Generated Answer (Raw)")
                    st.caption("Check if the LLM output format is correct:")
                    for dtype, snippet in first_snippets.items():
                        with st.expander(f"First {dtype} answer"):
                            st.code(snippet, language="python")
                
                if debug_info.get("errors"):
                    st.markdown("### Errors")
                    for error in debug_info["errors"]:
                        st.error(error)
                
                st.markdown("### Troubleshooting Tips")
                st.markdown("""
                1. **Relax constraints**: Reduce `min_answer_length`, increase `similarity_threshold`
                2. **Disable aggressive filtering**: Uncheck the toggle
                3. **Check testcase_generation**: Answers must contain proper pytest code with:
                   - `\\`\\`\\`python` fenced code block
                   - `def test_` function
                   - At least 2 `assert` statements  
                   - pytest feature (parametrize, raises, or fixture)
                4. **Check debug.json** in artifacts folder for full details
                """)
        
        st.markdown("---")
        
        # Tabs for different outputs
        tab_plan, tab_dataset, tab_critique, tab_eval = st.tabs([
            "📋 Action Plan",
            "📊 Dataset",
            "🔍 Critique",
            "📈 Evaluation",
        ])
        
        # ---------------------------------------------------------------------
        # Action Plan Tab
        # ---------------------------------------------------------------------
        with tab_plan:
            st.markdown(results["action_plan"])
        
        # ---------------------------------------------------------------------
        # Dataset Tab
        # ---------------------------------------------------------------------
        with tab_dataset:
            dataset = results["dataset"]
            
            st.subheader("Dataset Summary")
            st.write(f"**Generation Method**: {dataset.generation_method}")
            st.write(f"**LLM Provider**: {dataset.llm_provider or 'N/A'}")
            st.write(dataset.project_summary)
            
            st.markdown("---")
            
            # Dataset selector
            dataset_names = [ds.type for ds in dataset.datasets]
            selected_dataset = st.selectbox(
                "Select dataset type to view",
                options=dataset_names,
            )
            
            # Find selected dataset
            selected_ds = next(
                (ds for ds in dataset.datasets if ds.type == selected_dataset),
                None
            )
            
            if selected_ds:
                st.write(f"**Items**: {len(selected_ds.items)}")
                if selected_ds.intents:
                    st.write(f"**Intents**: {', '.join(selected_ds.intents)}")
                
                # Show items
                for i, item in enumerate(selected_ds.items):
                    with st.expander(f"Item {i+1}: {item.question[:60]}..."):
                        st.markdown("**Question:**")
                        st.write(item.question)
                        
                        st.markdown("**Answer:**")
                        st.markdown(item.answer)
                        
                        st.markdown("**Metadata:**")
                        st.json(item.metadata)
            
            st.markdown("---")
            
            # Download buttons for all formats
            st.subheader("📥 Export Formats")
            st.caption("Download in various fine-tuning formats")
            
            # JSON format
            dataset_json = json.dumps(
                dataset.model_dump(mode="json"),
                indent=2,
            )
            
            # Generate JSONL formats on-the-fly
            import io
            
            # QA JSONL
            qa_buffer = io.StringIO()
            for ds in dataset.datasets:
                for item in ds.items:
                    record = {
                        "question": item.question,
                        "answer": item.answer,
                        "metadata": {**item.metadata, "dataset_type": ds.type},
                    }
                    qa_buffer.write(json.dumps(record, ensure_ascii=False) + "\n")
            qa_jsonl = qa_buffer.getvalue()
            
            # Instruct JSONL
            instruct_buffer = io.StringIO()
            for ds in dataset.datasets:
                for item in ds.items:
                    record = {
                        "instruction": item.question,
                        "input": "",
                        "output": item.answer,
                        "metadata": {**item.metadata, "dataset_type": ds.type},
                    }
                    instruct_buffer.write(json.dumps(record, ensure_ascii=False) + "\n")
            instruct_jsonl = instruct_buffer.getvalue()
            
            # Chat JSONL
            chat_buffer = io.StringIO()
            for ds in dataset.datasets:
                for item in ds.items:
                    record = {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": item.question},
                            {"role": "assistant", "content": item.answer},
                        ],
                        "metadata": {**item.metadata, "dataset_type": ds.type},
                    }
                    chat_buffer.write(json.dumps(record, ensure_ascii=False) + "\n")
            chat_jsonl = chat_buffer.getvalue()
            
            # Download buttons in columns
            dl_col1, dl_col2 = st.columns(2)
            
            with dl_col1:
                st.download_button(
                    label="📄 dataset.json",
                    data=dataset_json,
                    file_name="dataset.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.download_button(
                    label="📝 dataset_qa.jsonl",
                    data=qa_jsonl,
                    file_name="dataset_qa.jsonl",
                    mime="application/jsonl",
                    use_container_width=True,
                    help="Simple Q&A format",
                )
            
            with dl_col2:
                st.download_button(
                    label="🎓 dataset_instruct.jsonl",
                    data=instruct_jsonl,
                    file_name="dataset_instruct.jsonl",
                    mime="application/jsonl",
                    use_container_width=True,
                    help="Alpaca-style instruction format",
                )
                st.download_button(
                    label="💬 dataset_chat.jsonl",
                    data=chat_jsonl,
                    file_name="dataset_chat.jsonl",
                    mime="application/jsonl",
                    use_container_width=True,
                    help="OpenAI chat format",
                )
            
            # Golden set download (if available)
            output_path = Path(results["output_path"])
            golden_path = output_path / "golden_set.jsonl"
            if golden_path.exists():
                with open(golden_path, "r", encoding="utf-8") as f:
                    golden_jsonl = f.read()
                st.download_button(
                    label="⭐ golden_set.jsonl",
                    data=golden_jsonl,
                    file_name="golden_set.jsonl",
                    mime="application/jsonl",
                    use_container_width=True,
                    help="Curated top items for evaluation",
                )
        
        # ---------------------------------------------------------------------
        # Critique Tab
        # ---------------------------------------------------------------------
        with tab_critique:
            critiques = results["critiques"]
            
            st.subheader("Self-Critique Results")
            st.caption("The critic agent reviews generated content for quality issues.")
            
            for dtype, critique in critiques.items():
                with st.expander(f"**{dtype}** — {critique.quality_assessment.upper()}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Rejected Items", len(critique.reject_indices))
                    with col2:
                        st.metric("Duplicate Pairs", len(critique.duplicate_pairs))
                    with col3:
                        st.metric("Low Quality", len(critique.low_quality_indices))
                    
                    if critique.improvement_notes:
                        st.markdown("**Improvement Notes:**")
                        for note in critique.improvement_notes:
                            st.write(f"- {note}")
                    else:
                        st.success("No major issues found.")
        
        # ---------------------------------------------------------------------
        # Evaluation Tab
        # ---------------------------------------------------------------------
        with tab_eval:
            evaluation = results["evaluation"]
            
            st.subheader("Quality Evaluation")
            
            # Overall rating with visual indicator
            rating = evaluation.overall_rating
            if rating >= 70:
                st.success(f"**Overall Rating: {rating:.1f}/100** — Good quality")
            elif rating >= 50:
                st.warning(f"**Overall Rating: {rating:.1f}/100** — Acceptable, room for improvement")
            else:
                st.error(f"**Overall Rating: {rating:.1f}/100** — Needs improvement")
            
            st.markdown("---")
            
            # Per-dataset scores
            st.subheader("Dataset Scores")
            
            for eval_ds in evaluation.dataset_evaluations:
                with st.expander(f"**{eval_ds.dataset_type}** — Uniqueness: {eval_ds.uniqueness_score:.1f}", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Uniqueness", f"{eval_ds.uniqueness_score:.1f}")
                    with col2:
                        st.metric("Lexical", f"{eval_ds.lexical_score:.1f}")
                    with col3:
                        st.metric("Structural", f"{eval_ds.structural_score:.1f}")
                    with col4:
                        st.metric("Conceptual", f"{eval_ds.conceptual_score:.1f}")
                    
                    st.write(f"**Items**: {eval_ds.item_count}")
                    st.write(f"**Avg Question Length**: {eval_ds.avg_question_length:.0f} chars")
                    st.write(f"**Avg Answer Length**: {eval_ds.avg_answer_length:.0f} chars")
            
            st.markdown("---")
            
            # Health Metrics
            st.subheader("Health Metrics")
            
            hm = evaluation.health_metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Answer Length", f"{hm.avg_answer_length:.0f} chars")
            with col2:
                st.metric("Items with Code", f"{hm.items_with_code} ({hm.items_with_code_pct:.1f}%)")
            with col3:
                st.metric("Intent Coverage", f"{hm.intent_coverage_score:.1f}%")
            
            if hm.difficulty_distribution:
                st.write("**Difficulty Distribution:**")
                diff_cols = st.columns(len(hm.difficulty_distribution))
                for i, (diff, count) in enumerate(hm.difficulty_distribution.items()):
                    with diff_cols[i]:
                        st.metric(diff.capitalize(), count)
            
            st.markdown("---")
            
            # Warnings
            if evaluation.warnings:
                st.subheader("⚠️ Warnings")
                for warning in evaluation.warnings:
                    st.warning(warning)
            
            # LLM Feedback
            if evaluation.llm_feedback:
                st.subheader("LLM Feedback")
                st.info(evaluation.llm_feedback)
            
            # Feedback list
            st.subheader("Recommendations")
            for line in evaluation.feedback:
                if line.strip():
                    st.write(line)
            
            st.markdown("---")
            
            # Download evaluation
            eval_json = json.dumps(
                evaluation.model_dump(mode="json"),
                indent=2,
                default=str,
            )
            st.download_button(
                label="⬇️ Download evaluation.json",
                data=eval_json,
                file_name="evaluation.json",
                mime="application/json",
            )
        
        # ---------------------------------------------------------------------
        # Artifacts Path
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.caption(f"📁 Artifacts saved to: `{results['output_path']}`")


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
footer_text = "Finetune Agent v2.0 | Built for the fine-tuning community"
if st.session_state.llm_provider == "ollama":
    footer_text += f" | Ollama: {st.session_state.ollama_model}"
st.caption(footer_text)
st.caption("This tool does NOT train models — it generates training datasets.")
