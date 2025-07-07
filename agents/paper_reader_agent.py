import os
import sys
import json
import requests
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_info, log_error

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def call_openrouter_api(messages, retries=3, backoff_factor=1):
    try:
        if not OPENROUTER_API_KEY:
            log_error("OPENROUTER_API_KEY environment variable is not set")
            raise ValueError("API key missing")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "qwen/qwq-32b:free",  # Your specified model
            "messages": messages
        }

        for attempt in range(retries):
            try:
                response = requests.post(url, headers=headers, data=json.dumps(data))
                response.raise_for_status()
                result = response.json()

                # Log full response for debugging
                log_info(f"API Response: {json.dumps(result, indent=2)}")

                if "choices" not in result:
                    log_error(f"Expected 'choices' in response, got: {result}")
                    raise KeyError("'choices' not found in API response")

                return result["choices"][0]["message"]["content"]

            except requests.exceptions.HTTPError as http_err:
                if response.status_code == 429:
                    sleep_time = backoff_factor * (2 ** attempt)
                    log_info(f"Rate limit hit, retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    log_error(f"HTTP Error: {http_err}")
                    raise
            except KeyError as key_err:
                log_error(f"API Response Missing Key: {key_err}")
                raise
            except Exception as e:
                log_error(f"API Request Error: {str(e)}")
                raise

        raise Exception("Max retries exceeded")

    except Exception as e:
        log_error(f"API Request Failed: {str(e)}")
        raise

def run_paper_reader_agent():
    try:
        log_info("Starting Paper Reader Agent...")

        # Load input paper content
        paper_input_path = "data/paper_input.txt"
        if not os.path.exists(paper_input_path):
            log_error(f"Input file '{paper_input_path}' does not exist")
            raise FileNotFoundError(f"Input file '{paper_input_path}' not found")

        with open(paper_input_path, "r", encoding="utf-8") as f:
            paper_content = f.read()

        if not paper_content.strip():
            log_error(f"Input file '{paper_input_path}' is empty")
            raise ValueError("Input file is empty")

        # Send request to LLM
        response = call_openrouter_api([
            {
                "role": "user",
                "content": paper_content
            }
        ])

        # Log response for confirmation
        log_info("LLM Response received.")

        # Save response
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/paper_summary.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)

        log_info(f"Paper summary saved to {output_path}.")

    except Exception as e:
        log_error(f"Paper Reader Agent Error: {str(e)}")
        raise

if __name__ == "__main__":
    run_paper_reader_agent()