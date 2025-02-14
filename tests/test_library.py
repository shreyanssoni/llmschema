import json
import logging
import pytest
from llmschema.core import generate_response, _extract_json
from llmschema.schema_manager import SchemaManager
from llmschema.exceptions import LLMValidationError
from pydantic import BaseModel
from typing import List, Optional
import ollama

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

# Sample Schema for Testing
class ResponseSchema(BaseModel):
    text: str
    confidence: float
    alternatives: Optional[List[str]] = None

# Set the test schema
SchemaManager.set_schema(ResponseSchema)

def test_generate_response_valid_json():
    """Test if generate_response returns a properly structured JSON output using actual Ollama model."""
    result = generate_response("llama3.2:1b", "Say hello")
    
    assert isinstance(result, dict)
    assert "text" in result
    assert "confidence" in result

def test_generate_response_invalid_json_retries(caplog):
    """Test if generate_response retries when LLM returns invalid JSON."""
    caplog.set_level(logging.WARNING)

    try:
        result = generate_response("llama3.2:1b", "Say something weird", max_retries=1)
        assert isinstance(result, dict)  # Should still return a valid dict after retries
    except LLMValidationError:
        assert "Retrying..." in caplog.text

def test_generate_response_fails_after_retries():
    """Test if generate_response raises LLMValidationError after max retries."""
    with pytest.raises(LLMValidationError, match="Failed to obtain a valid JSON response"):
        generate_response("llama3.2:1b", "Say random gibberish.", max_retries=1)

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

def test_generate_response_logs_errors(caplog):
    """Ensure that errors are logged when the model returns bad JSON."""
    caplog.set_level(logging.ERROR)

    try:
        generate_response("llama3.2:1b", "Give me nonsense output")
    except LLMValidationError:
        assert "JSON decoding failed" in caplog.text

def test_generate_response_validates_schema():
    """Ensure the response strictly follows the user-defined schema."""
    result = generate_response("llama3.2:1b", "Provide a structured response")
    assert "text" in result
    assert isinstance(result["text"], str)
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
