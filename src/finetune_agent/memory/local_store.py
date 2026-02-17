"""Local JSON-based memory store implementation."""

import json
from datetime import datetime
from pathlib import Path

from finetune_agent.memory.store import MemoryStore
from finetune_agent.schemas import RunSummary, UserProfile


class LocalMemoryStore(MemoryStore):
    """Local JSON file-based memory storage."""
    
    def __init__(self, storage_path: str | Path | None = None):
        """Initialize local storage.
        
        Args:
            storage_path: Path to the JSON file. Defaults to artifacts/memory.json
        """
        if storage_path is None:
            storage_path = Path("artifacts") / "memory.json"
        self._path = Path(storage_path)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self) -> None:
        """Ensure the storage file and directory exist."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write_data({"profile": None, "runs": []})
    
    def _read_data(self) -> dict:
        """Read data from the JSON file."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"profile": None, "runs": []}
    
    def _write_data(self, data: dict) -> None:
        """Write data to the JSON file."""
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_profile(self) -> UserProfile | None:
        """Get the user profile from local storage."""
        data = self._read_data()
        if data.get("profile"):
            return UserProfile.model_validate(data["profile"])
        return None
    
    def save_profile(self, profile: UserProfile) -> None:
        """Save the user profile to local storage."""
        profile.updated_at = datetime.now()
        data = self._read_data()
        data["profile"] = profile.model_dump(mode="json")
        self._write_data(data)
    
    def add_run(self, run: RunSummary) -> None:
        """Add a run summary to local storage."""
        data = self._read_data()
        runs = data.get("runs", [])
        runs.insert(0, run.model_dump(mode="json"))
        # Keep only the last 100 runs
        data["runs"] = runs[:100]
        self._write_data(data)
    
    def get_recent_runs(self, limit: int = 5) -> list[RunSummary]:
        """Get the most recent run summaries from local storage."""
        data = self._read_data()
        runs = data.get("runs", [])[:limit]
        return [RunSummary.model_validate(run) for run in runs]
    
    def clear(self) -> None:
        """Clear all stored data."""
        self._write_data({"profile": None, "runs": []})
