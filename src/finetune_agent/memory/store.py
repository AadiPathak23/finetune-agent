"""Memory store interface and factory."""

import os
from abc import ABC, abstractmethod

from finetune_agent.schemas import RunSummary, UserProfile


class MemoryStore(ABC):
    """Abstract base class for memory storage."""
    
    @abstractmethod
    def get_profile(self) -> UserProfile | None:
        """Get the user profile."""
        pass
    
    @abstractmethod
    def save_profile(self, profile: UserProfile) -> None:
        """Save the user profile."""
        pass
    
    @abstractmethod
    def add_run(self, run: RunSummary) -> None:
        """Add a run summary to history."""
        pass
    
    @abstractmethod
    def get_recent_runs(self, limit: int = 5) -> list[RunSummary]:
        """Get the most recent run summaries."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored data."""
        pass


def get_memory_store() -> MemoryStore:
    """Factory function to get the appropriate memory store.
    
    Returns Redis store if REDIS_URL is set, otherwise falls back to local JSON.
    """
    redis_url = os.environ.get("REDIS_URL")
    
    if redis_url:
        try:
            from finetune_agent.memory.redis_store import RedisMemoryStore
            store = RedisMemoryStore(redis_url)
            # Test connection
            store._client.ping()
            return store
        except Exception:
            # Fall back to local storage if Redis fails
            pass
    
    from finetune_agent.memory.local_store import LocalMemoryStore
    return LocalMemoryStore()
