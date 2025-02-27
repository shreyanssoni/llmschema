import requests
from ollama import chat
from ollama import ChatResponse
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel
from llmschema import SchemaManager
import re, json
from google import genai

class LLMClient:
    """Handles multiple LLM providers (DeepSeek, Gemini, Ollama)."""

    def __init__(self, provider: str, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize LLM client.

        :param provider: LLM provider ("deepseek", "gemini", or "ollama")
        :param model: Model name (for Ollama, DeepSeek)
        :param api_key: API key (required for DeepSeek & Gemini)
        """
        self.provider = provider.lower()
        self.model = model or ("deepseek-chat" if provider == "deepseek" else "deepseek-r1")
        self.api_key = api_key

    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Fetch LLM response based on provider."""
        try:
            if self.provider == "deepseek":
                return self._call_deepseek(prompt)
            elif self.provider == "gemini":
                return self._call_gemini(prompt)
            elif self.provider == "ollama":
                return self._call_ollama(prompt)
            else:
                return {"error": f"Unsupported provider: {self.provider}"}

        except Exception as e:
            return {"error": f"Error fetching LLM response: {e}"}

    def _call_deepseek(self, prompt: str) -> Dict[str, Any]:
        """Call DeepSeek API."""
        if not self.api_key:
            return {"error": "DeepSeek API key is required."}

        # url = "https://api.deepseek.com/v1/chat/completions"
        # headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        # payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}

        try:
            response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "deepseek/deepseek-r1-distill-llama-70b:free",
                "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
                ],
            })
            )
            
            response = response.json()
            return response['choices'][0]['message']['content']
        
        except Exception as e:
            print("Error calling deepseek api", e)
            return {"error": f"Error calling DeepSeek API: {e}"}

    def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        """Calls the Gemini API using the official Python library, with fallback."""
        if not self.api_key:
            return {"error": "Gemini API key is required."}
        
        try:
            client = genai.Client(api_key=self.api_key)
        
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )
            
            return response.text

        except Exception as e:
            # Fallback to the REST API if the Python library fails.
            print(f"Error using Python library: {e}. Falling back to REST API.")
            return e
        
    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama (runs locally)."""
    
        response = chat(model=self.model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])
        
        response_content = response['message']['content']
        
        json_pattern = re.search(r"```json\n(.*?)\n```", response_content, re.DOTALL)

        if json_pattern:
            json_str = json_pattern.group(1)  # Extract JSON inside backticks
            return json_str
        else:
            json_str = response_content  # Assume it's already JSON
            return json_str

    def structured_response(self, prompt: str) -> Dict[str, Any]:
        """
        Fetch response and validate it against the user-defined schema.
        """
        raw_response = self.get_response(prompt)

        # Retrieve schema set by user
        schema = SchemaManager.get_schema()
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                validated_response = schema(**raw_response)
                return validated_response.dict()
            except Exception as e:
                return {"error": f"Schema validation failed: {e}", "raw_response": raw_response}

        return {"error": "Invalid schema format. Must be a Pydantic model."}
