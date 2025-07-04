import os
import sys
import json

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_info, log_error

def run_notifier():
    try:
        log_info("Starting notifier agent...")

        # Load mutation log
        with open("experiments/results/mutation_log.json", "r") as f:
            mutation_log = json.load(f)
        log_info("Loaded mutation log successfully.")

        # Simulate notification (dummy print)
        message = f"""
📢 Notification:
Latest mutation completed successfully.
Mutation Type: {mutation_log['mutation_type']}
New Parameter: {mutation_log['new_parameter']}
New Value: {mutation_log['new_value']}
Previous Accuracy: {mutation_log['previous_accuracy']}
"""
        print(message.strip())
        log_info("Notification sent (simulated).")

    except Exception as e:
        log_error(f"Notifier Agent Error: {str(e)}")


if __name__ == "__main__":
    run_notifier()
