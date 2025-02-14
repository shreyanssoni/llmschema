# llmschema

`llmschema` is a Python library that ensures structured and validated responses from LLMs (Large Language Models) like Ollama, OpenAI, and Gemini by enforcing user-defined **Pydantic schemas**. It abstracts model-specific quirks and guarantees responses in a **safe, predictable, and JSON-compliant format**.

## 🚀 Features

✅ **Enforces Pydantic schema** on LLM responses  
✅ **Works with multiple LLM providers** (Ollama, OpenAI, Gemini, etc.)  
✅ **Handles malformed JSON responses gracefully**  
✅ **Easy integration** into existing applications  
✅ **Modular & scalable design**

---

## 📦 Installation

Install `llmschema` via pip:

```sh
pip install llmschema
```

---

## 🛠 Usage

### **1️⃣ Define a Schema**
```python
from pydantic import BaseModel
from llmschema import SchemaManager, generate_response

class MyResponseSchema(BaseModel):
    text: str
    confidence: float

SchemaManager.set_schema(MyResponseSchema)
```

### **2️⃣ Generate a Response from an LLM**
```python
response = generate_response("mistral", "Summarize the latest AI news")
print(response)  # Output will follow MyResponseSchema format
```

### **3️⃣ Handling Errors**
```python
from llmschema import LLMValidationError

try:
    response = generate_response("gemini", "Give me a JSON response")
except LLMValidationError as e:
    print("Invalid response:", e)
```

---

## ⚙️ Supported LLMs
`llmschema` is designed to work with different LLM providers:
- ✅ **Ollama** (Mistral, Llama, etc.)
- ✅ **OpenAI** (GPT models)
- ✅ **Gemini** (Google's LLM)

More integrations coming soon!

---

## ✅ Handling Non-JSON Responses
If an LLM outputs **invalid JSON**, `llmschema` will:
1. **Try to extract JSON** using regex.
2. **Log warnings** for malformed responses.
3. **Raise an error** if parsing fails completely.

---

## 🧪 Running Tests
To test the library locally:
```sh
pytest tests/
```

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🤝 Contributing
Contributions are welcome! Feel free to submit issues and PRs on GitHub.

GitHub Repo: [https://github.com/yourusername/llmschema](https://github.com/yourusername/llmschema)

