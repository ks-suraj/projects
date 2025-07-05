import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_info, log_error

def run_experiment():
    try:
        log_info("Starting experiment runner agent...")

        # Default hyperparameters
        hyperparameters = {
            "learning_rate": 0.001,
            "batch_size": 16,
            "optimizer": "SGD",
            "epochs": 5
        }

        # Check for mutation hyperparameters
        mutation_file = "experiments/results/mutation_log.json"
        if os.path.exists(mutation_file):
            with open(mutation_file, "r") as f:
                mutation = json.load(f)
            log_info("Loaded hyperparameters from mutation_log.json.")
            # If it's additive mutation, load all parameters
            if mutation.get("mutation_type") == "additive":
                hyperparameters.update(mutation["parameters"])
            else:
                # Single parameter tweak
                hyperparameters[mutation["new_parameter"]] = mutation["new_value"]
        else:
            log_info("No mutation log found. Using default hyperparameters.")

        # Simulate training with current hyperparameters
        log_info(f"Using hyperparameters: {hyperparameters}")
        log_info("Simulating training...")

        # Dummy training result
        experiment_result = {
            "accuracy": 47.95,  # Dummy
            "loss": 0.52,
            "hyperparameters": hyperparameters
        }

        # Save experiment results
        os.makedirs("experiments/results", exist_ok=True)
        with open("experiments/results/experiment_result.json", "w") as f:
            json.dump(experiment_result, f, indent=4)
        log_info("Experiment completed and results saved.")

    except Exception as e:
        log_error(f"Experiment Runner Agent Error: {str(e)}")


if __name__ == "__main__":
    run_experiment()
