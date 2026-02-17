"""Base LLM client interface."""

import json
from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Abstract base class for LLM clients.
    
    All LLM providers must implement this interface to ensure
    consistent behavior across different backends.
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider (e.g., 'openai', 'mock')."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text completion from a prompt.
        
        Args:
            prompt: The input prompt to complete
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_json(
        self, 
        prompt: str, 
        schema: dict[str, Any] | None = None,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        """Generate a JSON response from a prompt.
        
        Args:
            prompt: The input prompt (should request JSON output)
            schema: Optional JSON schema for validation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed JSON response as a dictionary
            
        Raises:
            ValueError: If the response cannot be parsed as JSON
        """
        pass
    
    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from a text response.
        
        Handles cases where the LLM wraps JSON in markdown code blocks.
        
        Args:
            text: Raw text that may contain JSON
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If no valid JSON can be extracted
        """
        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract from markdown code block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
        
        # Try to extract from generic code block
        if "```" in text:
            start = text.find("```") + 3
            # Skip language identifier if present
            newline = text.find("\n", start)
            if newline != -1:
                start = newline + 1
            end = text.find("```", start)
            if end != -1:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
        
        # Try to find JSON object in text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")
