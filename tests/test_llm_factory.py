"""Tests for the get_llm_client factory, focused on bring-your-own-key wiring."""

import pytest

from distillery.llm import get_llm_client
from distillery.llm.mock import MockLLMClient


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch):
    """Isolate from ambient provider/key env (and any local .env)."""
    for var in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_explicit_api_key_kwarg_is_honored():
    """A pasted key (kwarg) must build a real OpenAI client, not fall to mock.

    Regression: the factory previously decided the mock-fallback from the ENV
    key only, so a UI-provided api_key kwarg was ignored.
    """
    client = get_llm_client(
        provider="openai",
        api_key="sk-test-123",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
    )
    assert client.provider_name == "openai"
    assert client.model_name == "llama-3.1-8b-instant"


def test_openai_without_any_key_falls_back_to_mock():
    client = get_llm_client(provider="openai")
    assert isinstance(client, MockLLMClient)


def test_env_key_still_works(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
    client = get_llm_client(provider="openai", model="gpt-4o-mini")
    assert client.provider_name == "openai"
    assert client.model_name == "gpt-4o-mini"


def test_default_provider_is_mock():
    assert isinstance(get_llm_client(), MockLLMClient)
