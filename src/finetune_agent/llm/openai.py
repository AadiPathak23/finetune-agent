"""OpenAI-compatible LLM client."""

import json
import os
from typing import Any

import httpx

from .base import LLMClient


class OpenAIClient(LLMClient):
    """OpenAI API client implementation.
    
    Supports OpenAI and OpenAI-compatible APIs (e.g., Azure, local servers).
    
    Environment variables:
        OPENAI_API_KEY: API key for authentication
        OPENAI_BASE_URL: Optional custom base URL (default: https://api.openai.com/v1)
        OPENAI_MODEL: Model to use (default: gpt-4o-mini)
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            base_url: Custom base URL for API (or set OPENAI_BASE_URL env var)
            model: Model to use (or set OPENAI_MODEL env var)
        """
        self._api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("GROQ_API_KEY")  # allow Groq's OpenAI-compatible API
        )
        if not self._api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY (or GROQ_API_KEY) environment variable.")
        
        self._base_url = (
            base_url 
            or os.environ.get("OPENAI_BASE_URL") 
            or "https://api.openai.com/v1"
        )
        self._model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,  # LLM calls can be slow
        )
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text using OpenAI chat completions API."""
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        """Generate JSON using OpenAI with response_format enforcement."""
        # Add schema hint to prompt if provided
        enhanced_prompt = prompt
        if schema:
            enhanced_prompt += f"\n\nExpected JSON schema:\n```json\n{json.dumps(schema, indent=2)}\n```"
        
        # Request JSON response format
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "user", "content": enhanced_prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        return self._extract_json(content)
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
