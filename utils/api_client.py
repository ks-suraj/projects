import os
import sys
import requests
from dotenv import load_dotenv

# Add project root to sys.path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_info, log_error

# Load API key
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
        "model": "qwen/qwen3-32b:free",  # Change model here if needed
        "messages": [{"role": "user", "content": prompt}]
    }

    log_info("Sending request to OpenRouter API...")
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        log_error(f"API Error: {response.status_code} - {response.text}")
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    result = response.json()
    log_info("Received response from OpenRouter API.")
    return result["choices"][0]["message"]["content"]

if __name__ == "__main__":
    prompt = "Summarize the key points from the latest machine learning research papers."
    try:
        reply = query_openrouter(prompt)
        print("📝 OpenRouter Response:\n", reply)
    except Exception as e:
        log_error(f"API request failed: {e}")
