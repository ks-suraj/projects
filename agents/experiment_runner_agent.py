import os
import json
import datetime
import random
import sys

# Fix Import Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_info, log_error

def run_experiment():
    try:
        log_info("Starting Experiment Runner Agent...")

        # Load hyperparameters (mutation-aware)
        hyperparameters = {
            "learning_rate": 0.001,
            "batch_size": 16,
            "optimizer": "SGD",
            "epochs": 5
        }

        mutation_path = "experiments/results/mutation_log.json"
        if os.path.exists(mutation_path):
            with open(mutation_path, "r", encoding="utf-8") as f:
                mutation = json.load(f)
            if mutation.get("mutation_type") == "additive":
                hyperparameters.update(mutation["parameters"])
            log_info("Loaded mutated hyperparameters.")
        else:
            log_info("No mutation log found. Using default hyperparameters.")

        log_info(f"Running with hyperparameters: {hyperparameters}")

        # ✅ Simulate experiment (introduce random variations for realism)
        base_acc = 47.0
        acc_variation = random.uniform(-2.0, 2.0)
        accuracy = round(base_acc + acc_variation, 2)
        loss = round(random.uniform(0.4, 0.7), 2)

        experiment_result = {
            "accuracy": accuracy,
            "loss": loss,
            "hyperparameters": hyperparameters
        }

        # ✅ Save experiment result
        os.makedirs("experiments/results", exist_ok=True)
        with open("experiments/results/experiment_result.json", "w", encoding="utf-8") as f:
            json.dump(experiment_result, f, indent=4, ensure_ascii=False)

        log_info("Experiment result saved.")

        # ✅ Track Generations for graphing
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

        log_info(f"Logged generation {generation_entry['generation']}.")

    except Exception as e:
        log_error(f"Experiment Runner Agent Error: {str(e)}")

if __name__ == "__main__":
    run_experiment()
