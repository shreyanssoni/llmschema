import json
import logging
import ollama
import re
from .schema_manager import SchemaManager
from .exceptions import LLMValidationError
from pydantic import BaseModel
from typing import Optional, get_origin, get_args, Union

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_structured_prompt(schema, user_prompt):
    """
    Generates a structured prompt ensuring the AI provides a valid JSON response strictly following the schema.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema_details = schema.model_json_schema()
    elif isinstance(schema, dict):
        schema_details = schema
    else:
        raise ValueError("Invalid schema format. Must be a Pydantic model or a JSON schema dictionary.")
    
    properties = schema_details.get("properties", {})
    required_fields = set(schema_details.get("required", []))
    
    formatted_schema = {}
    for key, details in properties.items():
        field_description = details.get("description", f"A value for {key}.")
        field_type = details.get("type", "unknown")
        is_required = key in required_fields
        
        formatted_schema[key] = f"Give value for this. It's description: {field_description}, required: {is_required}. Strictly follow this type: {field_type}" 
        
    structured_prompt = f"""
    You are an AI assistant. Your task is to generate a response in **valid JSON format** that strictly follows the given schema.
    
    **Schema to Follow:**
    {json.dumps(formatted_schema, indent=2)}

    **User Prompt:** {user_prompt}
    """.strip()
    return structured_prompt

def generate_response(model: str, prompt: str, max_retries: int = 2):
    """
    Generates a structured response from an Ollama model while ensuring it follows the user-defined schema.
    """
    schema = SchemaManager.get_schema()
    structured_prompt = generate_structured_prompt(schema, prompt)
    
    for attempt in range(max_retries):
        logger.info(f"Attempt {attempt+1}: Sending request to Ollama model '{model}'")
        
        try:
            response = ollama.chat(model=model, messages=[{"role": "user", "content": structured_prompt}])
            response_text = response["message"]["content"]                
            response_json = _extract_json(response_text)
            validated_response = validate_response(response_json, schema)
            logger.info("Response successfully validated.")
            return validated_response
        
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON response. Retrying... (Attempt {attempt+1}/{max_retries})")
        except LLMValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    logger.error("Max retries reached. Raising error.")
    raise LLMValidationError("Failed to obtain a valid JSON response after retries.")

def _extract_json(response_text: str) -> dict:
    """Extracts valid JSON from a response text, handling common LLM quirks."""
    try:
        response_text = response_text.strip()
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1)
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed: {e}")
        raise

def validate_response(response_json: dict, schema) -> dict:
    """Validates the LLM-generated JSON response against the given schema."""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema_fields = schema.model_fields
    elif isinstance(schema, dict):
        schema_fields = schema.get("properties", {})
    else:
        raise ValueError("Invalid schema format. Must be a Pydantic model or a JSON schema dictionary.")
    
    validated_response = {}
    errors = []
    
    for field_name, field_info in schema_fields.items():
        annotation = getattr(field_info, "annotation", None)
        is_optional = get_origin(annotation) is Union and type(None) in get_args(annotation)
        field_value = response_json.get(field_name, None)
        
        if not is_optional and (field_value is None or (isinstance(field_value, str) and field_value.strip() == "")):
            errors.append(f"Missing or null value for required field: '{field_name}'")
        
        validated_response[field_name] = field_value
    
    if errors:
        raise LLMValidationError(
            message="Schema validation failed. Invalid or missing fields detected.",
            errors=errors,
            response_json=response_json
        )
    
    logger.info(f"Validated Response: {validated_response}")
    return validated_response
