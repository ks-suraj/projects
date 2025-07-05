import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_info, log_error

def run_experiment():
    try:
        log_info("Starting experiment runner agent...")

        # Load hyperparameters from saved config
        config_path = "experiments/results/next_experiment_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                experiment_config = json.load(f)
            log_info(f"Loaded hyperparameters: {experiment_config}")
        else:
            experiment_config = {
                "learning_rate": 0.001,
                "epochs": 10
            }
            log_info("Using default hyperparameters.")

        # Simulate training using hyperparameters
        log_info("Simulating training...")
        training_result = {
            "accuracy": 0.47 + 0.02,  # Dummy improvement
            "loss": 0.52 - 0.01,
            "epochs": experiment_config.get("epochs", 10)
        }

        # Save experiment results
        os.makedirs("experiments/results", exist_ok=True)
        with open("experiments/results/experiment_result.json", "w") as f:
            json.dump(training_result, f, indent=4)
        log_info("Experiment completed and results saved.")

    except Exception as e:
        log_error(f"Experiment Runner Agent Error: {str(e)}")


if __name__ == "__main__":
    run_experiment()
