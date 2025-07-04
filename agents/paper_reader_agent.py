import os
import requests
from dotenv import load_dotenv

# Load API key from .env for local development
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("API key missing! Please set OPENROUTER_API_KEY in .env or GitHub Secrets.")

def query_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral/mistral-7b-instruct",  # You can replace with any available model
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

if __name__ == "__main__":
    prompt = "Summarize the key points from the latest machine learning research papers."
    reply = query_openrouter(prompt)
    print("📝 OpenRouter Response:\n", reply)
