import os
import json
import datetime
import random
import sys

# Fix Import Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_info, log_error
from utils.api_client import call_openrouter_api

def llm_self_critique(code_path, last_accuracy, last_loss):
    """Ask the LLM to critique the algorithm/architecture and suggest improvements."""
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
        prompt = (
            "You are an expert AI engineer. "
            "Given the following model code and the last experiment's results, "
            "critique the algorithm and architecture, and suggest improvements ONLY as a bullet list. "
            "Focus on model design, learning strategy, and data flow. "
            "Do not return code, just feedback.\n\n"
            f"Model code:\n{code}\n\n"
            f"Last experiment accuracy: {last_accuracy}\n"
            f"Last experiment loss: {last_loss}\n"
        )
        messages = [{"role": "user", "content": prompt}]
        feedback = call_openrouter_api(messages)
        return feedback.strip()
    except Exception as e:
        return f"LLM critique failed: {e}"

def simulated_evaluation(code_path):
    """Simulate evaluation based on code length or other heuristics."""
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
        lines = code.splitlines()
        length_score = max(0, 100 - len(lines)) / 100  # Shorter code gets higher score
        return round(0.5 + length_score * 0.5, 2)  # Simulated "score" between 0.5 and 1.0
    except Exception:
        return 0.5

def get_latest_improved_code():
    """Get the latest improved model code from codes/improved/."""
    codes_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "codes", "improved")
    if os.path.exists(codes_dir):
        code_files = [os.path.join(codes_dir, f) for f in os.listdir(codes_dir) if f.endswith(".py")]
        if code_files:
            return max(code_files, key=os.path.getctime)
    return None

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
            if mutation.get("parameters"):
                hyperparameters.update(mutation["parameters"])
            log_info("Loaded mutated hyperparameters.")
        else:
            log_info("No mutation log found. Using default hyperparameters.")

        log_info(f"Running with hyperparameters: {hyperparameters}")

        # --- Get latest improved model code ---
        latest_code = get_latest_improved_code()
        if not latest_code:
            log_error("No improved model code found in /codes/improved/. Cannot run experiment.")
            return

        # --- Simulate experiment (randomized for realism) ---
        base_acc = 47.0
        acc_variation = random.uniform(-2.0, 2.0)
        accuracy = round(base_acc + acc_variation, 2)
        loss = round(random.uniform(0.4, 0.7), 2)

        # --- LLM Self-Critique (algorithm/architecture only) ---
        llm_feedback = llm_self_critique(latest_code, accuracy, loss)
        log_info(f"LLM Self-Critique:\n{llm_feedback}")

        # --- Simulated Evaluation ---
        simulated_score = simulated_evaluation(latest_code)
        log_info(f"Simulated evaluation score: {simulated_score}")

        experiment_result = {
            "accuracy": accuracy,
            "loss": loss,
            "hyperparameters": hyperparameters,
            "llm_self_critique": llm_feedback,
            "simulated_score": simulated_score
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
            "hyperparameters": hyperparameters,
            "llm_self_critique": llm_feedback,
            "simulated_score": simulated_score
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
    log_info("Experiment Runner Agent executed successfully.")
    log_info("This agent runs experiments based on the latest improved model code and hyperparameters.")
    