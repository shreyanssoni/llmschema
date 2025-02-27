from llmschema.core import generate_response
from llmschema.schema_manager import SchemaManager
from pydantic import BaseModel, Field

# Constants: Define providers, their respective models, and API keys

PROVIDERS = {
    "deepseek": {"model": "deepseek-r1", "api_key": "sk-or-v1-e"},
    "gemini": {"model": "gemini-2.0-flash", "api_key": "AIz"}
}

# Define a sample response schema using Pydantic
class SampleLLMResponseSchema(BaseModel):
    text: str = Field(..., description="Generated response text")
    confidence: float = Field(..., description="Confidence score of the response")
    source: str = Field(..., description="Source of information, if applicable")

def main():
    """Generate and print LLM responses using multiple providers with a structured schema."""

    # Convert Pydantic schema to JSON and set it in SchemaManager
    response_schema = SampleLLMResponseSchema.model_json_schema()
    SchemaManager.set_schema(response_schema)

    # Load and print the active schema
    schema = SchemaManager.get_schema()
    print("🔹 Using Response Schema:", schema)

    # Sample prompt
    prompt = "What is the capital of France?"
    
    for provider, details in PROVIDERS.items():
        model = details["model"]
        api_key = details["api_key"]
        
        print(f"\n🚀 Sending Prompt to {provider.upper()} (Model: {model})")
        
        # Call generate_response function
        response = generate_response(provider=provider, model=model, api_key=api_key, prompt=prompt)

        # Print response
        print(f"\n🔹 {provider.upper()} Response Output:")
        print(response)

if __name__ == "__main__":
    main()
