import os
import sys
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_info, log_error
from utils.api_client import call_openrouter_api

def run_model_designer():
    try:
        log_info("Starting Model Designer Agent...")

        # Load paper summary from outputs
        with open("outputs/paper_summary.txt", "r", encoding="utf-8") as f:
            paper_summary = f.read()

        # Prepare prompt
        prompt = f"""
        Given this paper summary:
        \"\"\"{paper_summary}\"\"\"

        Design a deep learning model based on the paper.
        Return ONLY a valid JSON object describing:
        - 'summary': Brief description of the model (in max 100 words).
        - 'design': PyTorch-like pseudocode (as string).
        - 'key_points': Key design notes (as string).

        Only return the JSON object. No additional text.
        """

        # Prepare messages for OpenRouter API
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Call LLM API with messages
        llm_response = call_openrouter_api(messages)
        print("LLM RAW Response:", llm_response)  # Debugging

        # Parse LLM response as JSON
        try:
            model_design = json.loads(llm_response)
        except json.JSONDecodeError as e:
            log_error(f"Failed to decode LLM response: {e}")
            raise ValueError("LLM returned invalid JSON.") from e

        # Save the model design
        os.makedirs("models/generated_models", exist_ok=True)
        with open("models/generated_models/generated_model.json", "w", encoding="utf-8") as f:
            json.dump(model_design, f, indent=4, ensure_ascii=False)

        log_info("Model design saved to models/generated_models/generated_model.json")

    except Exception as e:
        log_error(f"Model Designer Agent Error: {e}")

if __name__ == "__main__":
    run_model_designer()
