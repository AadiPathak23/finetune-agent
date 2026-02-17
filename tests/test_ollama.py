"""Tests for the Ollama LLM client.

Uses mocked httpx to test without actual Ollama server.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from finetune_agent.llm.ollama import OllamaClient, OllamaConnectionError


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client."""
    with patch("finetune_agent.llm.ollama.httpx.Client") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        yield client_instance


class TestOllamaClientInit:
    """Tests for OllamaClient initialization."""
    
    def test_default_values(self, mock_httpx_client):
        """Test that default host and model are used."""
        client = OllamaClient()
        
        assert client.host == "http://localhost:11434"
        assert client.model_name == "qwen2.5-coder"
        assert client.provider_name == "ollama"
    
    def test_custom_host_and_model(self, mock_httpx_client):
        """Test that custom host and model can be set."""
        client = OllamaClient(
            host="http://custom-host:12345",
            model="llama3.2",
        )
        
        assert client.host == "http://custom-host:12345"
        assert client.model_name == "llama3.2"
    
    def test_host_trailing_slash_removed(self, mock_httpx_client):
        """Test that trailing slash is removed from host."""
        client = OllamaClient(host="http://localhost:11434/")
        
        assert client.host == "http://localhost:11434"
    
    def test_env_vars_override(self, mock_httpx_client, monkeypatch):
        """Test that environment variables are used."""
        monkeypatch.setenv("OLLAMA_HOST", "http://env-host:9999")
        monkeypatch.setenv("OLLAMA_MODEL", "mistral")
        
        client = OllamaClient()
        
        assert client.host == "http://env-host:9999"
        assert client.model_name == "mistral"


class TestBuildGenerateRequest:
    """Tests for request building."""
    
    def test_basic_request_structure(self, mock_httpx_client):
        """Test that basic request has correct structure."""
        client = OllamaClient()
        
        request = client._build_generate_request("Hello, world!")
        
        assert request["model"] == "qwen2.5-coder"
        assert request["prompt"] == "Hello, world!"
        assert request["stream"] is False
        assert "options" in request
        assert request["options"]["num_predict"] == 2000
        assert request["options"]["temperature"] == 0.7
    
    def test_custom_max_tokens(self, mock_httpx_client):
        """Test custom max_tokens parameter."""
        client = OllamaClient()
        
        request = client._build_generate_request("Test", max_tokens=500)
        
        assert request["options"]["num_predict"] == 500
    
    def test_custom_temperature(self, mock_httpx_client):
        """Test custom temperature parameter."""
        client = OllamaClient()
        
        request = client._build_generate_request("Test", temperature=0.2)
        
        assert request["options"]["temperature"] == 0.2
    
    def test_json_format_flag(self, mock_httpx_client):
        """Test that json_format adds format field."""
        client = OllamaClient()
        
        request = client._build_generate_request("Test", json_format=True)
        
        assert request["format"] == "json"
    
    def test_no_json_format_by_default(self, mock_httpx_client):
        """Test that format is not added by default."""
        client = OllamaClient()
        
        request = client._build_generate_request("Test")
        
        assert "format" not in request


class TestGenerate:
    """Tests for the generate method."""
    
    def test_successful_generate(self, mock_httpx_client):
        """Test successful text generation."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {"response": "Generated text here"}
        mock_response.raise_for_status = Mock()
        mock_httpx_client.post.return_value = mock_response
        
        client = OllamaClient()
        result = client.generate("Tell me a joke")
        
        assert result == "Generated text here"
        mock_httpx_client.post.assert_called_once()
        
        # Verify the URL and payload
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/generate"
        payload = call_args[1]["json"]
        assert payload["prompt"] == "Tell me a joke"
        assert payload["model"] == "qwen2.5-coder"
    
    def test_generate_with_custom_max_tokens(self, mock_httpx_client):
        """Test generate with custom max_tokens."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "Result"}
        mock_response.raise_for_status = Mock()
        mock_httpx_client.post.return_value = mock_response
        
        client = OllamaClient()
        client.generate("Test prompt", max_tokens=100)
        
        payload = mock_httpx_client.post.call_args[1]["json"]
        assert payload["options"]["num_predict"] == 100


class TestGenerateJson:
    """Tests for the generate_json method."""
    
    def test_successful_json_generation(self, mock_httpx_client):
        """Test successful JSON generation."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"items": [{"question": "Q1", "answer": "A1"}]}'
        }
        mock_response.raise_for_status = Mock()
        mock_httpx_client.post.return_value = mock_response
        
        client = OllamaClient()
        result = client.generate_json("Generate JSON")
        
        assert result == {"items": [{"question": "Q1", "answer": "A1"}]}
    
    def test_json_format_requested(self, mock_httpx_client):
        """Test that JSON format is requested from Ollama."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "{}"}
        mock_response.raise_for_status = Mock()
        mock_httpx_client.post.return_value = mock_response
        
        client = OllamaClient()
        client.generate_json("Generate JSON")
        
        payload = mock_httpx_client.post.call_args[1]["json"]
        assert payload["format"] == "json"
    
    def test_schema_added_to_prompt(self, mock_httpx_client):
        """Test that schema is added to prompt."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "{}"}
        mock_response.raise_for_status = Mock()
        mock_httpx_client.post.return_value = mock_response
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        client = OllamaClient()
        client.generate_json("Generate", schema=schema)
        
        payload = mock_httpx_client.post.call_args[1]["json"]
        assert "Expected JSON schema" in payload["prompt"]
        assert '"name"' in payload["prompt"]
    
    def test_extracts_json_from_code_block(self, mock_httpx_client):
        """Test that JSON is extracted from markdown code blocks."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '```json\n{"result": "success"}\n```'
        }
        mock_response.raise_for_status = Mock()
        mock_httpx_client.post.return_value = mock_response
        
        client = OllamaClient()
        result = client.generate_json("Generate")
        
        assert result == {"result": "success"}


class TestCheckConnection:
    """Tests for the check_connection method."""
    
    def test_connection_success_model_found(self, mock_httpx_client):
        """Test successful connection with model available."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:latest"},
                {"name": "llama3.2:latest"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_httpx_client.get.return_value = mock_response
        
        client = OllamaClient()
        is_connected, message = client.check_connection()
        
        assert is_connected is True
        assert "qwen2.5-coder" in message
    
    def test_connection_success_model_not_found(self, mock_httpx_client):
        """Test connection works but model not available."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_httpx_client.get.return_value = mock_response
        
        client = OllamaClient()
        is_connected, message = client.check_connection()
        
        assert is_connected is False
        assert "not found" in message
        assert "ollama pull" in message
    
    def test_connection_failure(self, mock_httpx_client):
        """Test connection failure handling."""
        import httpx
        mock_httpx_client.get.side_effect = httpx.ConnectError("Connection refused")
        
        client = OllamaClient()
        is_connected, message = client.check_connection()
        
        assert is_connected is False
        assert "Cannot connect" in message
        assert "ollama serve" in message


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_connection_error_in_generate(self, mock_httpx_client):
        """Test that connection errors raise OllamaConnectionError."""
        import httpx
        mock_httpx_client.post.side_effect = httpx.ConnectError("Connection refused")
        
        client = OllamaClient()
        
        with pytest.raises(OllamaConnectionError) as exc_info:
            client.generate("Test")
        
        assert "Cannot connect" in str(exc_info.value)
    
    def test_model_not_found_error(self, mock_httpx_client):
        """Test that 404 error raises model not found message."""
        import httpx
        mock_response = Mock()
        mock_response.status_code = 404
        error = httpx.HTTPStatusError("Not found", request=Mock(), response=mock_response)
        mock_httpx_client.post.side_effect = error
        
        client = OllamaClient()
        
        with pytest.raises(OllamaConnectionError) as exc_info:
            client.generate("Test")
        
        assert "not found" in str(exc_info.value)


class TestLLMFactoryIntegration:
    """Tests for LLM factory with Ollama."""
    
    def test_factory_returns_ollama_client(self, mock_httpx_client, monkeypatch):
        """Test that factory returns OllamaClient for provider=ollama."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        
        from finetune_agent.llm import get_llm_client
        client = get_llm_client()
        
        assert client.provider_name == "ollama"
    
    def test_factory_with_kwargs(self, mock_httpx_client):
        """Test that factory passes kwargs to OllamaClient."""
        from finetune_agent.llm import get_llm_client
        
        client = get_llm_client(
            provider="ollama",
            host="http://custom:8080",
            model="custom-model",
        )
        
        assert client.host == "http://custom:8080"
        assert client.model_name == "custom-model"
