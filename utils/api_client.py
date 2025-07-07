import os
import json
import requests
import sys
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_info, log_error

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def call_openrouter_api(messages):
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "qwen/qwq-32b:free",
            "messages": messages
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        log_error(f"API Request Error: {e}")
        raise e

if __name__ == "__main__":
    # Test Case
    sample_messages = [
        {"role": "user", "content": "Hello! What is your purpose?"}
    ]
    print(call_openrouter_api(sample_messages))
