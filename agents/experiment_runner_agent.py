import sys
import os
import json

# Fix import path BEFORE any imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_info, log_error  # Now it works!
def run_experiment():
    try:
        log_info("Starting experiment runner agent...")

        # Load generated model design
        with open("models/generated_models/generated_model.json", "r") as f:
            model_design = json.load(f)
        log_info("Loaded model design successfully.")

        # Simulate training process (replace this with real training later)
        log_info("Simulating training on loaded model design...")
        training_result = {
            "accuracy": 0.85,
            "loss": 0.35,
            "epochs": 10
        }

        # Save dummy results
        os.makedirs("experiments/results", exist_ok=True)
        with open("experiments/results/experiment_result.json", "w") as f:
            json.dump(training_result, f, indent=4)
        log_info("Experiment completed successfully. Results saved.")
    
    except Exception as e:
        log_error(f"Experiment Runner Agent Error: {str(e)}")


if __name__ == "__main__":
    run_experiment()
