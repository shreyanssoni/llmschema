import json
import logging
import pytest
from unittest.mock import patch, MagicMock
from llmschema.core import generate_response, _extract_json
from llmschema.schema_manager import SchemaManager
from llmschema.exceptions import LLMValidationError
from pydantic import BaseModel
from typing import List, Optional

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


# Sample Schema for Testing
class ResponseSchema(BaseModel):
    text: str
    confidence: float
    alternatives: Optional[List[str]] = None


# Set the test schema
SchemaManager.set_schema(ResponseSchema)


@pytest.fixture
def mock_ollama():
    """Fixture to mock the Ollama chat API response."""
    with patch("llmschema.core.ollama.chat") as mock_chat:
        yield mock_chat


def test_generate_response_valid_json(mock_ollama):
    """Test if generate_response returns a properly structured JSON output."""
    mock_response = {
        "message": {"content": json.dumps({"text": "Hello", "confidence": 0.95, "alternatives": ["Hi", "Hey"]})}
    }
    mock_ollama.return_value = mock_response

    result = generate_response("mistral", "Say hello")

    assert isinstance(result, dict)
    assert result["text"] == "Hello"
    assert result["confidence"] == 0.95
    assert "alternatives" in result


def test_generate_response_invalid_json_retries(mock_ollama, caplog):
    """Test if generate_response retries when LLM returns invalid JSON."""
    caplog.set_level(logging.WARNING)

    # First response is invalid, second response is correct
    mock_ollama.side_effect = [
        {"message": {"content": "This is not JSON"}},
        {"message": {"content": json.dumps({"text": "Hello", "confidence": 0.9})}}
    ]

    result = generate_response("mistral", "Say hello", max_retries=1)

    assert result["text"] == "Hello"
    assert result["confidence"] == 0.9
    assert "Retrying..." in caplog.text


def test_generate_response_fails_after_retries(mock_ollama):
    """Test if generate_response raises LLMValidationError after max retries."""
    mock_ollama.return_value = {"message": {"content": "This is not JSON"}}

    with pytest.raises(LLMValidationError, match="Failed to obtain a valid JSON response"):
        generate_response("mistral", "Say hello", max_retries=2)


def test_extract_json_valid():
    """Test if _extract_json correctly parses valid JSON."""
    valid_json_str = '{"text": "Hello", "confidence": 0.95}'
    parsed_json = _extract_json(valid_json_str)

    assert parsed_json["text"] == "Hello"
    assert parsed_json["confidence"] == 0.95


def test_extract_json_handles_code_blocks():
    """Test if _extract_json properly extracts JSON from markdown code blocks."""
    json_with_code_block = "```json\n{\"text\": \"Hello\", \"confidence\": 0.95}\n```"
    parsed_json = _extract_json(json_with_code_block)

    assert parsed_json["text"] == "Hello"
    assert parsed_json["confidence"] == 0.95


def test_extract_json_invalid():
    """Test if _extract_json raises an error for malformed JSON."""
    invalid_json_str = "Hello, this is not JSON"
    with pytest.raises(json.JSONDecodeError):
        _extract_json(invalid_json_str)


def test_generate_response_logs_errors(mock_ollama, caplog):
    """Ensure that errors are logged when the model returns bad JSON."""
    caplog.set_level(logging.ERROR)

    mock_ollama.return_value = {"message": {"content": "Invalid JSON"}}

    with pytest.raises(LLMValidationError):
        generate_response("mistral", "Say something")

    assert "JSON decoding failed" in caplog.text


def test_generate_response_validates_schema(mock_ollama):
    """Ensure the response strictly follows the user-defined schema."""
    # Missing required field "text"
    mock_ollama.return_value = {"message": {"content": json.dumps({"confidence": 0.95})}}

    with pytest.raises(Exception):
        generate_response("mistral", "Say something")


