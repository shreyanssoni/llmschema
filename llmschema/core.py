import json
import logging
import ollama
from .schema_manager import SchemaManager
from .exceptions import LLMValidationError

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
    
    schema = SchemaManager.get_schema()  # Get user-defined schema

    structured_prompt = f"""
    You are an AI assistant. Please respond in **strict JSON format** matching this structure:
    {json.dumps(schema.model_json_schema(), indent=2)}

    DO NOT include explanations or extra text. JUST return the JSON.

    User Prompt: {prompt}
    """

    for attempt in range(max_retries + 1):
        logger.info(f"Attempt {attempt+1}: Sending request to Ollama model '{model}'")
        
        try:
            # Call the Ollama model
            response = ollama.chat(model=model, messages=[{"role": "user", "content": structured_prompt}])
            response_text = response["message"]["content"]

            # Ensure the response is valid JSON
            response_json = _extract_json(response_text)

            # Validate using Pydantic schema
            validated_response = schema.model_validate(response_json)
            logger.info("Response successfully validated.")
            return validated_response.model_dump()

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
