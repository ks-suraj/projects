import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "qwen/qwen3-32b:free",
    "messages": [{"role": "user", "content": "Hello, what is AI?"}]
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    reply = response.json()["choices"][0]["message"]["content"]
    print("✅ API Test Successful!")
    print("AI Response:", reply)
else:
    print(f"❌ API Error {response.status_code}: {response.text}")
