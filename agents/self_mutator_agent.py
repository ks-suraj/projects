import os
import sys
import json
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_info, log_error
from utils.api_client import query_openrouter  # Assuming you're using this for LLM

def run_self_mutator():
    try:
        log_info("Starting self-mutator agent...")

        # Load past experiment results
        with open("experiments/results/experiment_result.json", "r") as f:
            experiment_result = json.load(f)
        log_info("Loaded experiment results successfully.")

        # Send result summary to OpenRouter for mutation advice
        prompt = f"""
Given these experiment results:
Accuracy: {experiment_result['accuracy']}
Loss: {experiment_result['loss']}

Suggest improved hyperparameters in JSON format.
"""
        llm_response = query_openrouter(prompt)
        log_info("Received mutation suggestion from LLM.")

        # Extract and save hyperparameters from LLM response
        extract_and_save_hyperparams(llm_response)

    except Exception as e:
        log_error(f"Self-Mutator Agent Error: {str(e)}")


def extract_and_save_hyperparams(llm_response):
    try:
        json_blocks = re.findall(r"\{[\s\S]*?\}", llm_response)
        if json_blocks:
            suggested_config = json.loads(json_blocks[-1])  # Take last JSON
            os.makedirs("experiments/results", exist_ok=True)
            with open("experiments/results/next_experiment_config.json", "w") as f:
                json.dump(suggested_config, f, indent=4)
            log_info("Saved new hyperparameters for next experiment.")
        else:
            log_error("No JSON config found in LLM response.")
    except Exception as e:
        log_error(f"Failed to parse LLM hyperparameters: {str(e)}")


if __name__ == "__main__":
    run_self_mutator()
