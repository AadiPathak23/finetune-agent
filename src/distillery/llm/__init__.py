"""LLM abstraction layer for Distillery."""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import LLMClient


def get_llm_client(
    provider: str | None = None,
    **kwargs,
) -> "LLMClient":
    """Factory function to get the appropriate LLM client.
    
    Uses LLM_PROVIDER environment variable to determine which client to use.
    Defaults to mock if no provider is specified or no API key is available.
    
    Environment variables:
        LLM_PROVIDER: "openai" | "ollama" | "mock" (default: "mock")
        OPENAI_API_KEY: Required for OpenAI provider
        OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
        OLLAMA_MODEL: Model for Ollama (default: qwen2.5-coder)
    
    Args:
        provider: Override for LLM_PROVIDER env var
        **kwargs: Additional arguments passed to the client constructor
    
    Returns:
        An LLMClient instance
    """
    provider = provider or os.environ.get("LLM_PROVIDER", "mock")
    provider = provider.lower()
    
    if provider == "openai":
        # Honor an explicit api_key kwarg first (e.g. a key pasted in the UI),
        # then fall back to env. Accept GROQ_API_KEY too so Groq's
        # OpenAI-compatible endpoint works (OPENAI_API_KEY takes precedence).
        api_key = (
            kwargs.pop("api_key", None)
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("GROQ_API_KEY")
        )
        if api_key:
            from .openai import OpenAIClient
            # base_url / model still flow through **kwargs to the client.
            return OpenAIClient(api_key=api_key, **kwargs)
        else:
            # Fall back to mock if no API key
            from .mock import MockLLMClient
            return MockLLMClient()
    
    elif provider == "ollama":
        from .ollama import OllamaClient
        return OllamaClient(**kwargs)
    
    else:
        # Default to mock
        from .mock import MockLLMClient
        return MockLLMClient(**kwargs)


def get_available_providers() -> list[dict[str, str]]:
    """Get list of available LLM providers with their status.
    
    Returns:
        List of dicts with 'name', 'status', and 'message' keys
    """
    providers = []
    
    # Mock is always available
    providers.append({
        "name": "mock",
        "status": "available",
        "message": "Mock LLM for testing (always available)",
    })
    
    # Check OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        providers.append({
            "name": "openai",
            "status": "available",
            "message": "OpenAI API (API key configured)",
        })
    else:
        providers.append({
            "name": "openai",
            "status": "unavailable",
            "message": "OpenAI API (set OPENAI_API_KEY)",
        })
    
    # Check Ollama
    try:
        from .ollama import OllamaClient
        client = OllamaClient()
        is_connected, message = client.check_connection()
        providers.append({
            "name": "ollama",
            "status": "available" if is_connected else "unavailable",
            "message": message,
        })
    except Exception as e:
        providers.append({
            "name": "ollama",
            "status": "unavailable",
            "message": f"Ollama not available: {e}",
        })
    
    return providers


__all__ = ["get_llm_client", "get_available_providers", "LLMClient"]
