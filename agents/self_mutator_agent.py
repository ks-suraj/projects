import os
import json
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_info, log_error

def run_self_mutator():
    try:
        log_info("Starting self-mutator agent...")

        generation_log_path = "experiments/logs/generation_log.json"
        if not os.path.exists(generation_log_path):
            log_info("No generation log found. Skipping mutation.")
            return

        with open(generation_log_path, "r") as f:
            generations = json.load(f)

        if len(generations) < 2:
            log_info("Not enough generations to perform adaptive mutation. Skipping.")
            return

        last_gen = generations[-1]
        prev_gen = generations[-2]

        # Compare last 2 generations
        last_acc = last_gen["accuracy"]
        prev_acc = prev_gen["accuracy"]

        # Mutate learning rate (example)
        lr = last_gen["hyperparameters"]["learning_rate"]

        if last_acc >= prev_acc:
            new_lr = max(lr * 0.9, 1e-5)  # If improving, reduce LR slowly
            mutation_note = "Decreased learning rate to fine-tune"
        else:
            new_lr = min(lr * 1.1, 0.1)  # If not improving, increase LR
            mutation_note = "Increased learning rate to escape local minima"

        mutation = {
            "mutation_type": "parameter_tweak",
            "previous_accuracy": last_acc,
            "new_parameter": "learning_rate",
            "new_value": round(new_lr, 6),
            "note": mutation_note
        }

        os.makedirs("experiments/results", exist_ok=True)
        with open("experiments/results/mutation_log.json", "w") as f:
            json.dump(mutation, f, indent=4)

        log_info(f"Applied mutation: {mutation_note} → New LR: {new_lr}")

    except Exception as e:
        log_error(f"Self-Mutator Agent Error: {str(e)}")

if __name__ == "__main__":
    run_self_mutator()
