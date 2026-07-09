"""Utility functions for the Distillery."""

import uuid
from datetime import datetime
from pathlib import Path


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return str(uuid.uuid4())[:8]


def get_timestamp() -> str:
    """Get a formatted timestamp for folder naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_output_dir(base_path: str = "artifacts") -> Path:
    """Create a timestamped output directory.
    
    Args:
        base_path: Base path for artifacts
        
    Returns:
        Path to the created directory
    """
    timestamp = get_timestamp()
    output_dir = Path(base_path) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_markdown(content: str, path: Path, filename: str = "action_plan.md") -> Path:
    """Save markdown content to a file.
    
    Args:
        content: Markdown content to save
        path: Directory to save to
        filename: Name of the file
        
    Returns:
        Path to the saved file
    """
    file_path = path / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path


def save_json(data: dict | list, path: Path, filename: str) -> Path:
    """Save JSON data to a file.
    
    Args:
        data: Data to save
        path: Directory to save to
        filename: Name of the file
        
    Returns:
        Path to the saved file
    """
    import json
    
    file_path = path / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return file_path


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
