"""Streamlit UI for Distillery.

Run with: streamlit run src/distillery/ui/app.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from distillery.agent import FinetuneAgent
from distillery.exporter import export_chat_jsonl, export_instruct_jsonl, export_qa_jsonl
from distillery.schemas import DatasetConstraints, ModelFamily, UserConstraints


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Distillery",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Visual Design System (custom CSS)
# =============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;450;500;600;700&display=swap');

:root {
  --brand: #cde04a;
  --brand-2: #8faa2a;
  --brand-soft: #e6efac;
  --glow: rgba(205, 224, 74, 0.12);
  --bg: #0a0b0a;
  --panel: #101210;
  --card: rgba(255, 255, 255, 0.035);
  --card-hover: rgba(205, 224, 74, 0.07);
  --field: rgba(255, 255, 255, 0.04);
  --border: rgba(255, 255, 255, 0.09);
  --border-strong: rgba(255, 255, 255, 0.16);
  --text: #eef1ea;
  --text-dim: #969b8e;
}

/* ============ Base / typography ============ */
html, body, .stApp, [class*="css"], input, textarea, button, select, [data-baseweb] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; color: var(--text); }
.stApp { color: var(--text); }
p, span, label, li, .stMarkdown { color: var(--text); }

/* Flat background — no ambient glow (sharp, minimalist) */
.stApp {
  background: var(--bg);
}

/* ============ Strip default Streamlit chrome ============ */
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"], [data-testid="stStatusWidget"] { display: none; }
[data-testid="stHeader"] { background: transparent; height: 0; }
#MainMenu, footer { display: none; }

/* Tighten the giant default top padding + constrain width */
.block-container, [data-testid="stMainBlockContainer"] {
  padding-top: 2.1rem !important; padding-bottom: 3rem !important; max-width: 1300px;
}

/* ============ Hero (demoted: single small positioning line) ============ */
.distillery-hero { padding: 2px 0 16px; display: flex; align-items: center; gap: 9px; }
.distillery-hero .kicker {
  font-size: 0.9rem; font-weight: 500; letter-spacing: 0.01em; color: var(--text-dim);
}
.distillery-hero .kicker b { color: var(--brand-soft); font-weight: 600; }
.distillery-hero .hmark { display: inline-flex; color: var(--brand-soft); }
.distillery-hero .hmark svg { width: 18px; height: 18px; }

/* ============ Unified eyebrow (bordered uppercase pill, used everywhere) === */
.section-eyebrow {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase;
  color: var(--brand-soft);
  border: 1px solid var(--border); background: rgba(205,224,74,0.08);
  padding: 5px 11px; border-radius: 999px; margin: 6px 0 2px;
}
.section-eyebrow svg { width: 14px; height: 14px; }

/* ============ Section headings ============ */
[data-testid="stHeading"] h2 {
  font-size: 0.82rem !important; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.11em; color: var(--brand-soft);
  padding-bottom: 8px; margin-bottom: 6px; border-bottom: 1px solid var(--border);
}
[data-testid="stHeading"] h3 { font-size: 1.12rem !important; font-weight: 600; }

/* ============ Sidebar ============ */
section[data-testid="stSidebar"] {
  width: 408px !important;
  background: var(--panel);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div { padding-top: 0; }
[data-testid="stSidebarUserContent"] { padding: 1.15rem 1.25rem 2rem; }
/* Tighten vertical rhythm between sidebar widgets */
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap: 0.7rem; }
/* Sidebar brand lockup */
.sidebar-brand {
  display: flex; align-items: center; gap: 11px;
  padding: 2px 2px 15px; margin-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
.sidebar-brand .mark {
  display: inline-flex; align-items: center; justify-content: center;
  width: 40px; height: 40px; border-radius: 10px; color: var(--brand);
  background: rgba(205,224,74,0.08);
  border: 1px solid var(--border-strong);
}
.sidebar-brand .mark svg { width: 22px; height: 22px; }
.sidebar-brand .name {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 1.22rem; letter-spacing: -0.02em; line-height: 1.1;
  color: var(--text);
}
.sidebar-brand .sub {
  font-size: 0.72rem; color: var(--text-dim); margin-top: 1px; letter-spacing: 0.01em;
}
/* Sidebar section headers -> tidy control-panel labels */
section[data-testid="stSidebar"] [data-testid="stHeading"] h2 {
  margin-top: 10px !important;
}

/* ============ Cards / feature grid (empty state) ============ */
.empty-wrap { max-width: 940px; margin: 6px 0 0; }
.empty-panel {
  border: 1px solid var(--border); border-radius: 18px;
  background: var(--card);
  padding: 34px 34px 36px;
}
.empty-eyebrow {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--brand-soft);
  border: 1px solid var(--border); background: rgba(205,224,74,0.08);
  padding: 5px 11px; border-radius: 999px;
}
.empty-eyebrow svg { width: 13px; height: 13px; }
.empty-title {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 1.9rem; letter-spacing: -0.02em; line-height: 1.15;
  margin: 16px 0 8px; color: var(--text);
}
.empty-title .accent { color: var(--brand); }
.empty-sub { color: var(--text-dim); font-size: 1.02rem; line-height: 1.55; max-width: 620px; }

.feature-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 30px;
}
.feature-card {
  position: relative; overflow: hidden;
  border: 1px solid var(--border); border-radius: 16px;
  background: rgba(255,255,255,0.02);
  padding: 20px 18px 18px;
  transition: transform .18s ease, border-color .18s ease, background .18s ease, box-shadow .18s ease;
}
.feature-card:hover {
  transform: translateY(-2px); border-color: var(--border-strong);
  background: var(--card-hover);
}
.feature-card .step {
  position: absolute; top: 14px; right: 15px;
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.92rem;
  color: var(--brand-soft); opacity: 0.85;
}
.feature-card .ficon {
  width: 42px; height: 42px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  color: var(--brand);
  background: rgba(205,224,74,0.07);
  border: 1px solid var(--border-strong);
  margin-bottom: 15px;
}
.feature-card .ficon svg { width: 22px; height: 22px; }
.feature-card .ftitle {
  font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.02rem;
  color: var(--text); margin-bottom: 6px;
}
.feature-card .ftext { color: var(--text-dim); font-size: 0.83rem; line-height: 1.5; }
.empty-cta {
  display: flex; align-items: center; gap: 9px; margin-top: 28px;
  padding-top: 20px; border-top: 1px solid var(--border);
  font-size: 0.92rem; color: var(--text-dim);
}
.empty-cta svg { color: var(--brand-soft); flex-shrink: 0; }
.empty-cta b { color: var(--brand-soft); font-weight: 600; }
@media (max-width: 1150px) { .feature-grid { grid-template-columns: repeat(2, 1fr); } }

/* ============ Inputs (baseweb) ============ */
.stTextInput div[data-baseweb="input"], .stTextArea div[data-baseweb="textarea"],
.stNumberInput div[data-baseweb="input"],
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
  background: var(--field) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; transition: border-color .15s ease, box-shadow .15s ease;
}
.stTextInput input, .stTextArea textarea, .stNumberInput input { color: var(--text) !important; }
.stTextInput div[data-baseweb="input"]:focus-within, .stTextArea div[data-baseweb="textarea"]:focus-within,
.stNumberInput div[data-baseweb="input"]:focus-within,
.stSelectbox div[data-baseweb="select"]:focus-within > div,
.stMultiSelect div[data-baseweb="select"]:focus-within > div {
  border-color: var(--brand) !important; box-shadow: 0 0 0 3px var(--glow) !important;
}
/* Selectbox / multiselect popover + tags */
[data-baseweb="popover"] [role="listbox"] {
  background: var(--panel) !important; border: 1px solid var(--border-strong) !important; border-radius: 12px !important;
}
[data-baseweb="tag"] {
  background: rgba(205,224,74,0.12) !important; border: 1px solid rgba(205,224,74,0.30) !important;
  border-radius: 7px !important; color: var(--brand-soft) !important;
}
[data-baseweb="tag"] svg { fill: var(--brand-soft) !important; }

/* Slider */
.stSlider [data-baseweb="slider"] [role="slider"] { background: var(--brand) !important; border-color: var(--brand) !important; }
.stSlider [data-baseweb="slider"] > div > div > div { background: var(--brand) !important; }
/* Value is inlined into the label, so hide the floating bare thumb value */
.stSlider [data-testid="stSliderThumbValue"],
.stSlider [data-testid="stThumbValue"] { display: none !important; }

/* ============ Buttons ============ */
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
  border-radius: 11px; border: 1px solid var(--border); font-weight: 600;
  background: var(--field); color: var(--text);
  transition: transform .14s ease, border-color .14s ease, box-shadow .14s ease, background .14s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--brand); background: var(--card-hover); transform: translateY(-1px);
}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"],
div[data-testid="stFormSubmitButton"] > button {
  background: var(--brand) !important;
  border: none !important; color: #0a0b0a !important;
  font-weight: 700 !important; font-size: 1.0rem; padding: 0.55rem 1rem;
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
  filter: brightness(1.06); transform: translateY(-1px);
}

/* ============ Metric cards ============ */
[data-testid="stMetric"] {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 15px 17px;
  transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-2px); border-color: var(--border-strong); background: var(--card-hover);
}
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #fff; }
[data-testid="stMetricLabel"] p { opacity: .65; text-transform: uppercase; font-size: .68rem; letter-spacing: .07em; font-weight: 600; }

/* ============ Alerts (unified lime/neutral — no default Streamlit blue) === */
[data-testid="stAlert"] {
  border-radius: 12px !important;
  border: 1px solid rgba(205,224,74,0.32) !important;
  background: var(--card) !important;
  backdrop-filter: blur(6px);
}
[data-testid="stAlert"] > div,
[data-testid="stAlertContainer"],
[data-testid="stAlert"] [data-baseweb="notification"] {
  background: transparent !important;
  color: var(--text) !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] li,
[data-testid="stAlert"] div { color: var(--text) !important; }
[data-testid="stAlert"] svg { fill: var(--brand-soft) !important; color: var(--brand-soft) !important; }
[data-testid="stAlert"] code { color: #e6efac !important; }
/* Semantic left accent — keeps success/error legible while staying on-system */
[data-testid="stAlert"]:has(svg) { border-left-width: 3px !important; }

/* ============ Expanders ============ */
[data-testid="stExpander"] {
  border: 1px solid var(--border); border-radius: 14px; background: var(--card); overflow: hidden;
}
[data-testid="stExpander"] summary { padding: 12px 16px; font-weight: 600; }
[data-testid="stExpander"] summary:hover { color: var(--brand-soft); }

/* ============ Tabs ============ */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
  font-weight: 600; border-radius: 10px 10px 0 0; padding: 9px 18px; color: var(--text-dim);
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text); background: var(--card); }
.stTabs [aria-selected="true"] { color: #fff !important; background: var(--card); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--brand) !important; height: 2.5px; }

/* ============ Code ============ */
code { background: rgba(205, 224, 74, 0.14) !important; color: #e6efac !important; border-radius: 6px; padding: 1px 6px; font-size: 0.86em; }
[data-testid="stCode"], pre { border: 1px solid var(--border) !important; border-radius: 12px !important; }

/* ============ DataFrame / tables ============ */
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }

/* ============ Captions / dividers ============ */
[data-testid="stCaptionContainer"], small { color: var(--text-dim) !important; }
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* ============ Progress / spinner accent ============ */
.stProgress > div > div > div { background: linear-gradient(90deg, var(--brand), var(--brand-2)) !important; }
[data-testid="stSpinner"] i { border-top-color: var(--brand) !important; }

/* ============ Scrollbar ============ */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(205, 224, 74, 0.35); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(205, 224, 74, 0.55); }
::-webkit-scrollbar-track { background: transparent; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Icon set — one cohesive inline-SVG line-icon system (1.5px stroke,
# currentColor). No full-color emoji in any HTML surface.
# =============================================================================

def _svg(body: str, size: int = 24) -> str:
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
        f'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )


ICONS = {
    # Brand mark — laboratory flask (distillation)
    "flask": _svg(
        '<path d="M9 2h6"/>'
        '<path d="M9.5 2v6.2L4.7 16a3 3 0 0 0 2.6 4.6h9.4a3 3 0 0 0 2.6-4.6L14.5 8.2V2"/>'
        '<path d="M7.4 14h9.2"/>'
    ),
    # Plan — compass
    "compass": _svg(
        '<circle cx="12" cy="12" r="9"/>'
        '<path d="M15.6 8.4l-2.1 5.1-5.1 2.1 2.1-5.1z"/>'
    ),
    # Generate — gear
    "gear": _svg(
        '<circle cx="12" cy="12" r="3.2"/>'
        '<path d="M12 2.5v3M12 18.5v3M21.5 12h-3M5.5 12h-3'
        'M18.7 5.3l-2.1 2.1M7.4 16.6l-2.1 2.1M18.7 18.7l-2.1-2.1M7.4 7.4L5.3 5.3"/>'
    ),
    # Verify — shield with check
    "shield": _svg(
        '<path d="M12 3l7 3v5c0 4.4-3 7.6-7 9-4-1.4-7-4.6-7-9V6z"/>'
        '<path d="M9 12l2.2 2.2L15.2 10"/>'
    ),
    # Evaluate — bar chart
    "chart": _svg(
        '<path d="M3.5 20.5h17"/>'
        '<rect x="5.5" y="11" width="3.2" height="7" rx="1"/>'
        '<rect x="10.4" y="6" width="3.2" height="12" rx="1"/>'
        '<rect x="15.3" y="13" width="3.2" height="5" rx="1"/>'
    ),
    # LLM Provider — processor / chip
    "cpu": _svg(
        '<rect x="7" y="7" width="10" height="10" rx="2"/>'
        '<path d="M9.5 2v3M14.5 2v3M9.5 19v3M14.5 19v3'
        'M2 9.5h3M2 14.5h3M19 9.5h3M19 14.5h3"/>'
    ),
    # Configuration — sliders
    "sliders": _svg(
        '<path d="M5 4v5M5 13v7M12 4v9M12 17v3M19 4v3M19 11v9"/>'
        '<circle cx="5" cy="11" r="2"/><circle cx="12" cy="15" r="2"/>'
        '<circle cx="19" cy="9" r="2"/>'
    ),
    # Small diamond bullet for the pipeline eyebrow
    "diamond": _svg('<path d="M12 3.5l8.5 8.5-8.5 8.5L3.5 12z"/>', size=13),
    # Arrow (CTA)
    "arrow": _svg('<path d="M11 5l7 7-7 7"/><path d="M18 12H4"/>', size=15),
}


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
        st.session_state.ollama_model = "qwen2.5:7b-instruct"
    if "ollama_status" not in st.session_state:
        st.session_state.ollama_status = None
    # OpenAI-compatible (bring-your-own-key) settings — session-only, never
    # written to disk.
    if "openai_preset" not in st.session_state:
        st.session_state.openai_preset = "OpenAI"
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "openai_base_url" not in st.session_state:
        st.session_state.openai_base_url = "https://api.openai.com/v1"
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-4o-mini"
    if "openai_status" not in st.session_state:
        st.session_state.openai_status = None
    # Groq (cloud, OpenAI-compatible) — session-only key, never written to disk.
    if "groq_api_key" not in st.session_state:
        # Never pre-fill the env key into a UI field (it would be revealable).
        # An env key is used silently at runtime via the client factory instead.
        st.session_state.groq_api_key = ""
    if "groq_model" not in st.session_state:
        st.session_state.groq_model = "llama-3.3-70b-versatile"
    if "groq_status" not in st.session_state:
        st.session_state.groq_status = None


init_session_state()


# =============================================================================
# Helper Functions
# =============================================================================

def check_ollama_connection(host: str, model: str) -> tuple[bool, str]:
    """Check Ollama connection status."""
    try:
        from distillery.llm.ollama import OllamaClient
        client = OllamaClient(host=host, model=model)
        return client.check_connection()
    except Exception as e:
        return False, f"Error: {e}"


def check_openai_connection(api_key: str, base_url: str, model: str) -> tuple[bool, str]:
    """Check an OpenAI-compatible endpoint by making a tiny generation call."""
    from distillery.llm import get_llm_client
    try:
        client = get_llm_client(
            provider="openai",
            api_key=api_key or None,
            base_url=base_url or None,
            model=model or None,
        )
        if client.provider_name != "openai":
            return False, "No API key provided — would fall back to mock."
        client.generate("ping", max_tokens=5)
        return True, f"Connected — model '{client.model_name}' responded."
    except Exception as e:
        return False, f"Connection failed: {e}"


def get_llm_client_for_provider(
    provider: str,
    ollama_host: str = None,
    ollama_model: str = None,
    openai_api_key: str = None,
    openai_base_url: str = None,
    openai_model: str = None,
    groq_api_key: str = None,
    groq_model: str = None,
):
    """Get LLM client based on selected provider."""
    from distillery.llm import get_llm_client

    if provider == "ollama":
        return get_llm_client(
            provider="ollama",
            host=ollama_host or st.session_state.ollama_host,
            model=ollama_model or st.session_state.ollama_model,
        )
    elif provider == "groq":
        # Groq is OpenAI-compatible; reuse the OpenAI client with Groq's base URL.
        return get_llm_client(
            provider="openai",
            api_key=(groq_api_key or st.session_state.groq_api_key) or None,
            base_url="https://api.groq.com/openai/v1",
            model=(groq_model or st.session_state.groq_model) or None,
        )
    elif provider == "openai":
        return get_llm_client(
            provider="openai",
            api_key=(openai_api_key or st.session_state.openai_api_key) or None,
            base_url=(openai_base_url or st.session_state.openai_base_url) or None,
            model=(openai_model or st.session_state.openai_model) or None,
        )
    else:
        return get_llm_client(provider=provider)


# =============================================================================
# Header
# =============================================================================

st.markdown(
    f'<div class="distillery-hero">'
    f'<span class="hmark">{ICONS["flask"]}</span>'
    f'<span class="kicker">Agentic <b>fine-tuning dataset</b> generator</span>'
    f'</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# Sidebar: All configuration controls (control-panel layout)
# =============================================================================

with st.sidebar:
    st.markdown(
        f'<div class="sidebar-brand">'
        f'<div class="mark">{ICONS["flask"]}</div>'
        f'<div><div class="name">Distillery</div>'
        f'<div class="sub">Dataset control panel</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    # =========================================================================
    # LLM Provider Configuration
    # =========================================================================
    st.markdown(
        f'<div class="section-eyebrow">{ICONS["cpu"]}<span>LLM Provider</span></div>',
        unsafe_allow_html=True,
    )
    
    provider_options = ["groq", "openai", "ollama", "mock"]
    provider_labels = {
        "groq": "Groq (Cloud · free key)",
        "openai": "OpenAI-compatible",
        "ollama": "Ollama (Local)",
        "mock": "Mock (Testing)",
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
    
    elif selected_provider == "groq":
        with st.expander("⚡ Groq Settings", expanded=True):
            # If a key is configured in the environment, use it silently — never
            # render it in a field (a password input is still revealable, which
            # would expose the owner's key to anyone using the app).
            env_key_present = bool(os.getenv("GROQ_API_KEY"))
            if env_key_present:
                st.success(
                    "🔒 Using the Groq API key from the environment — "
                    "hidden and never displayed."
                )
            else:
                st.caption(
                    "Free, fast, OpenAI-compatible cloud inference — recommended. "
                    "[Get a free API key ↗](https://console.groq.com/keys)"
                )
                st.session_state.groq_api_key = st.text_input(
                    "Groq API key",
                    value=st.session_state.groq_api_key,
                    type="password",
                    placeholder="gsk_...",
                    help="Kept in this session only — never written to disk.",
                )
            groq_models = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
            ]
            cur_model = st.session_state.groq_model
            model_options = (
                groq_models if cur_model in groq_models else [cur_model] + groq_models
            )
            st.session_state.groq_model = st.selectbox(
                "Model",
                options=model_options,
                index=model_options.index(cur_model),
                help="70b-versatile = best quality; 8b-instant = fastest with the "
                "highest free-tier rate limits.",
            )

            if st.button("🔄 Test Connection", use_container_width=True):
                with st.spinner("Testing Groq connection..."):
                    st.session_state.groq_status = check_openai_connection(
                        st.session_state.groq_api_key or os.getenv("GROQ_API_KEY", ""),
                        "https://api.groq.com/openai/v1",
                        st.session_state.groq_model,
                    )

            if st.session_state.groq_status is not None:
                ok, message = st.session_state.groq_status
                (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {message}")

            if not env_key_present and not st.session_state.groq_api_key:
                st.caption("No key set yet — generation would fall back to mock.")

    elif selected_provider == "openai":
        with st.expander("🔑 OpenAI-compatible Settings", expanded=True):
            # Presets pre-fill base URL + a sensible default model. "Custom"
            # lets the user point at any OpenAI-compatible endpoint.
            presets = {
                "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
                "Groq": ("https://api.groq.com/openai/v1", "llama-3.1-8b-instant"),
                "Custom": (st.session_state.openai_base_url, st.session_state.openai_model),
            }
            preset_names = list(presets.keys())
            preset = st.selectbox(
                "Preset",
                options=preset_names,
                index=preset_names.index(st.session_state.openai_preset)
                if st.session_state.openai_preset in preset_names
                else 0,
                help="Pick a provider to pre-fill the base URL and model.",
            )
            # When the preset changes, refresh base URL + model to its defaults.
            if preset != st.session_state.openai_preset and preset != "Custom":
                st.session_state.openai_base_url, st.session_state.openai_model = presets[preset]
            st.session_state.openai_preset = preset

            st.session_state.openai_api_key = st.text_input(
                "API key",
                value=st.session_state.openai_api_key,
                type="password",
                help="Kept in this session only — never written to disk.",
            )
            st.session_state.openai_base_url = st.text_input(
                "Base URL",
                value=st.session_state.openai_base_url,
            )
            st.session_state.openai_model = st.text_input(
                "Model",
                value=st.session_state.openai_model,
                help="e.g. gpt-4o-mini, llama-3.1-8b-instant, llama-3.3-70b-versatile",
            )

            if st.button("🔄 Test Connection", use_container_width=True):
                with st.spinner("Testing connection..."):
                    st.session_state.openai_status = check_openai_connection(
                        st.session_state.openai_api_key,
                        st.session_state.openai_base_url,
                        st.session_state.openai_model,
                    )

            if st.session_state.openai_status is not None:
                ok, message = st.session_state.openai_status
                (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {message}")

            if not st.session_state.openai_api_key:
                st.caption("No key set yet — generation would fall back to mock.")

    elif selected_provider == "mock":
        st.info("Using mock LLM for testing — no API calls are made.")
    
    st.markdown("---")
    
    # =========================================================================
    # Dataset Configuration
    # =========================================================================
    st.markdown(
        f'<div class="section-eyebrow">{ICONS["sliders"]}<span>Configuration</span></div>',
        unsafe_allow_html=True,
    )
    
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
    
    # Q&A count — current value inlined into the label (no floating bare number)
    _qa_current = st.session_state.get("qa_per_type_slider", 10)
    qa_per_type = st.slider(
        f"Q&A pairs per type — {_qa_current}",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        key="qa_per_type_slider",
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

    # Correctness gate (runs generated tests locally)
    validate_generated_code = st.toggle(
        "✅ Validate generated code (runs tests)",
        value=False,
        help=(
            "For test-case datasets, statically check and execute self-contained "
            "generated tests, rejecting ones that fail. Runs LLM-generated code "
            "locally in a sandboxed subprocess — enable only if you trust the run."
        ),
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
                openai_api_key=st.session_state.openai_api_key if selected_provider == "openai" else None,
                openai_base_url=st.session_state.openai_base_url if selected_provider == "openai" else None,
                openai_model=st.session_state.openai_model if selected_provider == "openai" else None,
                groq_api_key=st.session_state.groq_api_key if selected_provider == "groq" else None,
                groq_model=st.session_state.groq_model if selected_provider == "groq" else None,
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
            validate_generated_code=validate_generated_code,
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
# Main canvas: Outputs (empty state or results dashboard)
# =============================================================================

with st.container():
    if st.session_state.results is None:
        # ---- Marketed empty state: pipeline explainer + feature cards ----
        # NOTE: kept as one line-free block so Streamlit's markdown parser does
        # not treat indented lines as a code block.
        empty_state_html = (
            '<div class="empty-wrap"><div class="empty-panel">'
            f'<span class="empty-eyebrow">{ICONS["diamond"]}<span>The pipeline</span></span>'
            '<div class="empty-title">Four agents, one <span class="accent">verified</span> dataset.</div>'
            '<div class="empty-sub">Four specialized agents turn one prompt into a dataset that '
            'proves itself by running the tests it writes.</div>'
            '<div class="feature-grid">'
            f'<div class="feature-card"><span class="step">01</span><div class="ficon">{ICONS["compass"]}</div>'
            '<div class="ftitle">Plan</div><div class="ftext">The Planner drafts a spec covering target '
            'model, dataset design and risks.</div></div>'
            f'<div class="feature-card"><span class="step">02</span><div class="ficon">{ICONS["gear"]}</div>'
            '<div class="ftitle">Generate</div><div class="ftext">The Generator produces diverse Q&amp;A '
            'pairs with difficulty and intent labels.</div></div>'
            f'<div class="feature-card"><span class="step">03</span><div class="ficon">{ICONS["shield"]}</div>'
            '<div class="ftitle">Verify</div><div class="ftext">The Critic flags weak pairs and runs the '
            'generated tests under pytest.</div></div>'
            f'<div class="feature-card"><span class="step">04</span><div class="ficon">{ICONS["chart"]}</div>'
            '<div class="ftitle">Evaluate</div><div class="ftext">The Evaluator scores uniqueness and '
            'correctness, then exports clean JSONL.</div></div>'
            '</div>'
            f'<div class="empty-cta">{ICONS["arrow"]}<span>Configure a run in the panel on the left, '
            'then press <b>Run Agent</b>.</span></div>'
            '</div></div>'
        )
        st.markdown(empty_state_html, unsafe_allow_html=True)

    else:
        st.header("Results")
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

        # Correctness-gate coverage — the headline "it actually verifies itself" stat
        vcov = debug_info.get("verification_coverage", {})
        if vcov:
            graded = sum(s["graded"] for s in vcov.values())
            executed = sum(s["executed"] for s in vcov.values())
            passed = sum(s["passed"] for s in vcov.values())
            rejected = sum(s["rejected"] for s in vcov.values())
            skipped = sum(s["skipped_external_deps"] for s in vcov.values())
            skipped_note = (
                f" · {skipped} skipped (external deps)" if skipped else ""
            )
            st.markdown(
                f"""
                <div style="
                  border:1px solid rgba(205,224,74,0.35);
                  background:linear-gradient(120deg, rgba(205,224,74,0.13), rgba(143,170,42,0.10));
                  border-radius:16px; padding:16px 18px; margin:6px 0 2px;
                  display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
                  <div style="font-size:1.7rem; line-height:1;">🔬</div>
                  <div style="flex:1; min-width:190px;">
                    <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                                font-size:1.04rem; color:#ece7ff;">
                      Self-verified — the pipeline ran its own tests
                    </div>
                    <div style="color:#a9a4c0; font-size:0.86rem; margin-top:3px;">
                      Correctness gate graded {graded} generated tests under
                      <code>pytest</code>{skipped_note}.
                    </div>
                  </div>
                  <div style="display:flex; gap:24px; text-align:center;">
                    <div>
                      <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                                  font-size:1.55rem; color:#cde04a;">{passed}</div>
                      <div style="font-size:.66rem; letter-spacing:.06em; text-transform:uppercase;
                                  color:#a9a4c0;">Passed</div>
                    </div>
                    <div>
                      <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                                  font-size:1.55rem; color:#e6efac;">{executed}</div>
                      <div style="font-size:.66rem; letter-spacing:.06em; text-transform:uppercase;
                                  color:#a9a4c0;">Executed</div>
                    </div>
                    <div>
                      <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                                  font-size:1.55rem; color:#f87171;">{rejected}</div>
                      <div style="font-size:.66rem; letter-spacing:.06em; text-transform:uppercase;
                                  color:#a9a4c0;">Rejected</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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

            st.caption(
                f"🎯 **Correctness (LLM-judge): {evaluation.correctness_score:.1f}/100** — "
                "faithfulness + usefulness of the answers. This is the *correctness* axis, "
                "distinct from the diversity scores below."
            )

            st.markdown("---")

            # Per-dataset scores
            st.subheader("Dataset Scores")

            for eval_ds in evaluation.dataset_evaluations:
                with st.expander(f"**{eval_ds.dataset_type}** — Uniqueness: {eval_ds.uniqueness_score:.1f}", expanded=True):
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("Uniqueness", f"{eval_ds.uniqueness_score:.1f}")
                    with col2:
                        st.metric("Lexical", f"{eval_ds.lexical_score:.1f}")
                    with col3:
                        st.metric("Structural", f"{eval_ds.structural_score:.1f}")
                    with col4:
                        st.metric("Conceptual", f"{eval_ds.conceptual_score:.1f}")
                    with col5:
                        st.metric("Correctness", f"{eval_ds.correctness_score:.1f}")
                    
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

footer_text = "Distillery v2.0 · Built for the fine-tuning community"
if st.session_state.llm_provider == "ollama":
    footer_text += f" · Ollama: {st.session_state.ollama_model}"
st.markdown(
    f"""
    <div style="margin-top:34px; padding-top:18px; border-top:1px solid var(--border);
                display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap;
                color:var(--text-dim); font-size:0.8rem;">
      <span>{footer_text}</span>
      <span>This tool does <b style="color:#e6efac;">not</b> train models — it generates training datasets.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
