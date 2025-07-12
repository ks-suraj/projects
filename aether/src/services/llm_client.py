from aether.config.secrets import OPENROUTER_API_KEY
import requests

class LLMClient:
    def __init__(self, model: str = "mistralai/mistral-nemo:free"):
        self.model = model
        self.api_key = OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1"

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"❌ LLM generation failed: {str(e)}")
            return f"LLM generation failed: {str(e)}"
