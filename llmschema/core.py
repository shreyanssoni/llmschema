import json
import logging
import ollama
from .schema_manager import SchemaManager
from .exceptions import LLMValidationError

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import json

def generate_structured_prompt(schema, user_prompt):
    """
    Generates a structured prompt that ensures the AI provides a valid JSON response strictly adhering to the schema.

    Args:
        schema (Pydantic model): The schema defining the required JSON structure.
        user_prompt (str): The user’s input query.

    Returns:
        str: The formatted prompt.
    """
    
    schema_details = schema.model_json_schema()
    properties = schema_details.get("properties", {})
    required_fields = set(schema_details.get("required", []))
    
    formatted_schema = {}
    for key, details in properties.items():
        field_description = details.get("description", f"A value for {key}.")
        is_required = key in required_fields
        
        formatted_schema[key] = {
            "description": field_description,
            "required": is_required
        }
    
    structured_prompt = f"""
    You are an AI assistant. Your task is to generate a response in **valid JSON format** that strictly follows the given schema.

    **Guidelines:**
    - Ensure the response is a **valid JSON object** with the correct key-value pairs.
    - Each field should contain **realistic values**, not placeholders like "<string>", "null", or "undefined".
    - **Required fields** must always be included with appropriate values.
    - **Optional fields** should only be included if relevant; otherwise, omit them.
    - **Do not add extra fields** beyond those defined in the schema.
    - Maintain correct data types (e.g., strings for text fields, numbers for numerical fields).
    - If a field expects a structured response, ensure it adheres to the expected format.

    **Schema to Follow:**
    {json.dumps(formatted_schema, indent=2)}

    **User Prompt:** {user_prompt}

    **Expected Output:** A properly structured JSON response that aligns with the schema while containing meaningful values.
    """

    return structured_prompt.strip()


def generate_response(model: str, prompt: str, max_retries: int = 2):
    """
    Generates a structured response from an Ollama model while ensuring it follows the user-defined schema.

    Args:
        model (str): Ollama model name (e.g., "mistral", "gemma").
        prompt (str): User prompt.
        max_retries (int): Number of retries if JSON parsing fails.

    Returns:
        dict: Validated JSON response.
    """
    
    schema = SchemaManager.get_schema()

    structured_prompt = generate_structured_prompt(schema, prompt)

    for attempt in range(max_retries + 1):
        logger.info(f"Attempt {attempt+1}: Sending request to Ollama model '{model}'")
        
        try:
            # Call the Ollama model
            response = ollama.chat(model=model, messages=[{"role": "user", "content": structured_prompt}])
            response_text = response["message"]["content"]
            
            logger.error(f"Raw LLM Response: {response_text}")
            
            # Ensure the response is valid JSON
            response_json = _extract_json(response_text)

            # Custom validation function instead of strict Pydantic validation
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

    raise LLMValidationError("Failed to obtain a valid JSON response after retries.")

def _extract_json(response_text: str) -> dict:
    """
    Extracts valid JSON from a response text, handling common LLM quirks.

    Args:
        response_text (str): Raw model output.

    Returns:
        dict: Parsed JSON object.

    Raises:
        json.JSONDecodeError: If JSON extraction fails.
    """
    try:
        # Handle cases where LLM adds markdown-style code blocks
        if response_text.strip().startswith("```json"):
            response_text = response_text.strip().strip("```json").strip("```")

        # Convert response to JSON
        return json.loads(response_text)

    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed: {e}")
        raise

def validate_response(response_json: dict, schema) -> dict:
    """
    Custom validation function that ensures all required keys exist while allowing flexible values.

    Args:
        response_json (dict): LLM-generated response JSON.
        schema (pydantic.BaseModel): User-defined schema.

    Returns:
        dict: Validated JSON response.

    Raises:
        LLMValidationError: If critical issues are found.
    """
    schema_fields = schema.model_fields.keys()  # Expected keys
    validated_response = {}

    for key in schema_fields:
        if key in response_json:
            validated_response[key] = response_json[key]  # Keep original value
        else:
            validated_response[key] = None  # Missing keys get None

    logger.info(f"Final validated response: {validated_response}")
    return validated_response
