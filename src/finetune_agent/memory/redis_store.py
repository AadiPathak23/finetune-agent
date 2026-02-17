"""Redis-based memory store implementation."""

import json
from datetime import datetime

import redis

from finetune_agent.memory.store import MemoryStore
from finetune_agent.schemas import RunSummary, UserProfile


class RedisMemoryStore(MemoryStore):
    """Redis-based memory storage."""
    
    PROFILE_KEY = "finetune_agent:profile"
    RUNS_KEY = "finetune_agent:runs"
    
    def __init__(self, redis_url: str):
        """Initialize Redis connection."""
        self._client = redis.from_url(redis_url, decode_responses=True)
    
    def get_profile(self) -> UserProfile | None:
        """Get the user profile from Redis."""
        data = self._client.get(self.PROFILE_KEY)
        if data:
            return UserProfile.model_validate_json(data)
        return None
    
    def save_profile(self, profile: UserProfile) -> None:
        """Save the user profile to Redis."""
        profile.updated_at = datetime.now()
        self._client.set(self.PROFILE_KEY, profile.model_dump_json())
    
    def add_run(self, run: RunSummary) -> None:
        """Add a run summary to Redis list."""
        self._client.lpush(self.RUNS_KEY, run.model_dump_json())
        # Keep only the last 100 runs
        self._client.ltrim(self.RUNS_KEY, 0, 99)
    
    def get_recent_runs(self, limit: int = 5) -> list[RunSummary]:
        """Get the most recent run summaries from Redis."""
        data = self._client.lrange(self.RUNS_KEY, 0, limit - 1)
        return [RunSummary.model_validate_json(item) for item in data]
    
    def clear(self) -> None:
        """Clear all stored data."""
        self._client.delete(self.PROFILE_KEY, self.RUNS_KEY)
