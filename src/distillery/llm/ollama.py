"""Ollama LLM client for local model inference.

Run Qwen2.5-Coder or other models locally for free using Ollama.

Setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model: ollama pull qwen2.5-coder
    3. Start server: ollama serve
    4. Set environment: LLM_PROVIDER=ollama

Environment variables:
    OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL: Model to use (default: qwen2.5-coder)
"""

import os
from typing import Any

import httpx

from .base import LLMClient


class OllamaConnectionError(Exception):
    """Raised when cannot connect to Ollama server."""
    pass


class OllamaClient(LLMClient):
    """Ollama API client for local LLM inference.
    
    Uses the Ollama REST API to generate text with locally-hosted models
    like Qwen2.5-Coder, Llama, Mistral, etc.
    
    Environment variables:
        OLLAMA_HOST: Server URL (default: http://localhost:11434)
        OLLAMA_MODEL: Model name (default: qwen2.5-coder)
    """
    
    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "qwen2.5-coder"
    
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float = 300.0,  # Longer timeout for local inference
    ):
        """Initialize the Ollama client.
        
        Args:
            host: Ollama server URL (or set OLLAMA_HOST env var)
            model: Model name to use (or set OLLAMA_MODEL env var)
            timeout: Request timeout in seconds (default: 300s for slow local inference)
        """
        self._host = (
            host 
            or os.environ.get("OLLAMA_HOST") 
            or self.DEFAULT_HOST
        )
        self._model = (
            model 
            or os.environ.get("OLLAMA_MODEL") 
            or self.DEFAULT_MODEL
        )
        self._timeout = timeout
        
        # Normalize host URL (remove trailing slash)
        self._host = self._host.rstrip("/")
        
        self._client = httpx.Client(
            timeout=self._timeout,
            headers={"Content-Type": "application/json"},
        )
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model
    
    @property
    def host(self) -> str:
        """Return the Ollama host URL."""
        return self._host
    
    def check_connection(self) -> tuple[bool, str]:
        """Check if Ollama server is running and model is available.
        
        Returns:
            Tuple of (is_connected, message)
        """
        try:
            # Check server is running
            response = self._client.get(f"{self._host}/api/tags")
            response.raise_for_status()
            
            # Check if model is available
            data = response.json()
            available_models = [m["name"] for m in data.get("models", [])]
            
            # Model names can be with or without tag (e.g., "qwen2.5-coder" or "qwen2.5-coder:latest")
            model_found = any(
                self._model in m or m.startswith(self._model) 
                for m in available_models
            )
            
            if not model_found:
                return False, (
                    f"Model '{self._model}' not found. "
                    f"Available models: {', '.join(available_models) or 'none'}. "
                    f"Run: ollama pull {self._model}"
                )
            
            return True, f"Connected to Ollama with model '{self._model}'"
            
        except httpx.ConnectError:
            return False, (
                f"Cannot connect to Ollama at {self._host}. "
                "Make sure Ollama is installed and running:\n"
                "1. Install Ollama: https://ollama.com/download\n"
                "2. Start server: ollama serve\n"
                "3. Pull model: ollama pull qwen2.5-coder"
            )
        except httpx.HTTPStatusError as e:
            return False, f"Ollama server error: {e}"
        except Exception as e:
            return False, f"Unexpected error connecting to Ollama: {e}"
    
    def _build_generate_request(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        json_format: bool = False,
    ) -> dict[str, Any]:
        """Build the request payload for Ollama /api/generate.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            json_format: Whether to request JSON output
            
        Returns:
            Request payload dictionary
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        
        if json_format:
            payload["format"] = "json"
        
        return payload
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text using Ollama /api/generate endpoint.
        
        Args:
            prompt: The input prompt to complete
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            OllamaConnectionError: If cannot connect to Ollama server
        """
        payload = self._build_generate_request(prompt, max_tokens)
        
        try:
            response = self._client.post(
                f"{self._host}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
            
        except httpx.ConnectError:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._host}. "
                "Make sure Ollama is running: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise OllamaConnectionError(
                    f"Model '{self._model}' not found. "
                    f"Run: ollama pull {self._model}"
                )
            raise OllamaConnectionError(f"Ollama API error: {e}")
    
    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        """Generate JSON using Ollama with format enforcement.
        
        Args:
            prompt: The input prompt (should request JSON output)
            schema: Optional JSON schema for validation hint
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed JSON response as a dictionary
            
        Raises:
            OllamaConnectionError: If cannot connect to Ollama server
            ValueError: If the response cannot be parsed as JSON
        """
        # Enhance prompt with JSON instructions
        enhanced_prompt = prompt
        if schema:
            import json as json_module
            enhanced_prompt += f"\n\nExpected JSON schema:\n```json\n{json_module.dumps(schema, indent=2)}\n```"
        
        enhanced_prompt += "\n\nRespond with valid JSON only. No additional text."
        
        payload = self._build_generate_request(
            enhanced_prompt, 
            max_tokens, 
            temperature=0.5,  # Lower temperature for more consistent JSON
            json_format=True,
        )
        
        try:
            response = self._client.post(
                f"{self._host}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            
            data = response.json()
            content = data.get("response", "")
            
            return self._extract_json(content)
            
        except httpx.ConnectError:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._host}. "
                "Make sure Ollama is running: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise OllamaConnectionError(
                    f"Model '{self._model}' not found. "
                    f"Run: ollama pull {self._model}"
                )
            raise OllamaConnectionError(f"Ollama API error: {e}")
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
