import os
import sys
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.api_client import call_openrouter_api
from utils.logger import log_info, log_error

def run_paper_reader_agent():
    try:
        log_info("Starting Paper Reader Agent...")

        # Load input paper content
        with open("data/paper_input.txt", "r", encoding="utf-8") as f:
            paper_content = f.read()

        # Send request to LLM
        response = call_openrouter_api([
            {
                "role": "user",
                "content": paper_content
            }
        ])

        log_info("LLM Response received.")

        # Save response
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/paper_summary.txt", "w", encoding="utf-8") as f:
            f.write(response)

        log_info("Paper summary saved to outputs/paper_summary.txt.")

    except Exception as e:
        log_error(f"Paper Reader Agent Error: {str(e)}")

if __name__ == "__main__":
    run_paper_reader_agent()
