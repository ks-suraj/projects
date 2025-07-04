import os
import sys
import json

# Fix the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_info, log_error

def run_self_mutator():
    try:
        log_info("Starting self-mutator agent...")

        # Load past experiment results
        with open("experiments/results/experiment_result.json", "r") as f:
            experiment_result = json.load(f)
        log_info("Loaded experiment results successfully.")

        # Simulate mutation logic based on results (this is dummy for now)
        log_info("Performing self-mutation based on experiment results...")
        mutation_details = {
            "mutation_type": "parameter_tweak",
            "previous_accuracy": experiment_result["accuracy"],
            "new_parameter": "learning_rate",
            "new_value": 0.001  # Dummy tweak
        }

        # Save mutation details
        os.makedirs("experiments/results", exist_ok=True)
        with open("experiments/results/mutation_log.json", "w") as f:
            json.dump(mutation_details, f, indent=4)
        log_info("Mutation completed successfully. Mutation details saved.")

    except Exception as e:
        log_error(f"Self-Mutator Agent Error: {str(e)}")


if __name__ == "__main__":
    run_self_mutator()
