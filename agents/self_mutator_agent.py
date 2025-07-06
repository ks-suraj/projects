import os
import sys
import json

# Setup project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_info, log_error
from utils.api_client import call_openrouter_api

def run_self_mutator():
    try:
        log_info("🔄 Starting Self-Mutator Agent (AI-powered)...")

        experiment_result_path = "experiments/results/experiment_result.json"
        if not os.path.exists(experiment_result_path):
            raise FileNotFoundError(f"{experiment_result_path} not found. Run experiment first.")

        # Load previous experiment results
        with open(experiment_result_path, "r", encoding="utf-8") as f:
            experiment_result = json.load(f)

        # Prepare message for LLM (Send previous input, hyperparameters & result)
        prompt = f"""
You are an AI-based optimizer.

Here is the previous experiment result:
Accuracy: {experiment_result['accuracy']}
Loss: {experiment_result['loss']}
Hyperparameters: {json.dumps(experiment_result['hyperparameters'], indent=2)}

Suggest improved hyperparameters for the next run. 
Return ONLY a JSON object with updated hyperparameters.
"""

        llm_response = call_openrouter_api([
            {"role": "user", "content": prompt}
        ])

        log_info("✅ LLM Response received.")
        log_info(f"LLM Response:\n{llm_response}")

        # Try parsing LLM's response as JSON, removing markdown code fences if present
        try:
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```"):
                # Remove code fences and optional language tag
                cleaned_response = cleaned_response.split("```")[1]
                cleaned_response = cleaned_response.strip()
            suggested_params = json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError("LLM did not return valid JSON. Please verify the response format.")

        # Save Mutation Log
        mutation_log = {
            "mutation_type": "parameter_tweak",
            "previous_accuracy": experiment_result["accuracy"],
            "parameters": suggested_params
        }

        os.makedirs("experiments/results", exist_ok=True)
        with open("experiments/results/mutation_log.json", "w", encoding="utf-8") as f:
            json.dump(mutation_log, f, indent=4, ensure_ascii=False)

        log_info("✅ New mutation log saved successfully.")

    except Exception as e:
        log_error(f"Self-Mutator Agent Error: {str(e)}")


if __name__ == "__main__":
    run_self_mutator()      
    log_info("Self Mutator Agent executed successfully.")
    log_info("This agent mutates hyperparameters based on previous experiment results.")
    