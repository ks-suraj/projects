import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()  # Load from .env

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def chat_with_openrouter(messages, model="qwen/qwen3-32b:free"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()
