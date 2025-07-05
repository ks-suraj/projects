import os
import json
import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_info, log_error

def simulate_accuracy_loss(hyperparameters):
    lr = hyperparameters.get("learning_rate", 0.001)
    batch_size = hyperparameters.get("batch_size", 16)
    
    # Simulate Accuracy: Inversely related to LR, mildly to batch size
    accuracy = 0.5 + 0.2 * (0.001 / lr) - 0.01 * (batch_size / 16)
    accuracy = min(max(accuracy, 0.0), 1.0)  # Clamp between 0-1
    accuracy = round(accuracy * 100, 2)  # Percent
    
    # Simulate Loss: Opposite to Accuracy
    loss = round(1.0 - (accuracy / 100), 4)
    
    return accuracy, loss

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
            if mutation.get("mutation_type") == "additive":
                hyperparameters.update(mutation["parameters"])
            else:
                hyperparameters[mutation["new_parameter"]] = mutation["new_value"]
        else:
            log_info("No mutation log found. Using default hyperparameters.")

        # Simulate Training Results
        accuracy, loss = simulate_accuracy_loss(hyperparameters)

        experiment_result = {
            "accuracy": accuracy,
            "loss": loss,
            "hyperparameters": hyperparameters
        }

        os.makedirs("experiments/results", exist_ok=True)
        with open("experiments/results/experiment_result.json", "w") as f:
            json.dump(experiment_result, f, indent=4)

        log_info(f"Experiment completed with Accuracy={accuracy}%, Loss={loss}. Results saved.")

        # ✅ Track generations
        generation_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "accuracy": accuracy,
            "loss": loss,
            "hyperparameters": hyperparameters
        }

        os.makedirs("experiments/logs", exist_ok=True)
        generation_log_path = "experiments/logs/generation_log.json"

        if os.path.exists(generation_log_path):
            with open(generation_log_path, "r", encoding="utf-8") as f:
                generation_log = json.load(f)
        else:
            generation_log = []

        generation_entry["generation"] = len(generation_log) + 1
        generation_log.append(generation_entry)

        with open(generation_log_path, "w", encoding="utf-8") as f:
            json.dump(generation_log, f, indent=4, ensure_ascii=False)

        log_info(f"Logged generation {generation_entry['generation']} to {generation_log_path}")

    except Exception as e:
        log_error(f"Experiment Runner Agent Error: {str(e)}")

if __name__ == "__main__":
    run_experiment()
