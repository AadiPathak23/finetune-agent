"""Finetune Agent - An agentic AI assistant for finetuning engineering."""

from pathlib import Path

# Load environment variables from a project-root .env file (if present) so
# users don't have to export shell vars on every run. Shell-set vars still take
# precedence (override=False). python-dotenv is optional: if it's not installed,
# fall back silently to whatever is already in the environment.
try:
    from dotenv import load_dotenv

    # In the src/ layout this file is src/finetune_agent/__init__.py, so the
    # project root is two parents up. Fall back to a cwd-based search otherwise.
    _root_env = Path(__file__).resolve().parents[2] / ".env"
    if _root_env.exists():
        load_dotenv(_root_env)
    else:
        load_dotenv()
except ImportError:
    pass

__version__ = "0.1.0"
