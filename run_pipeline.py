import subprocess
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_info, log_error

def run_agent(agent_path):
    try:
        log_info(f"Running {agent_path}...")
        subprocess.run([sys.executable, agent_path], check=True)
        log_info(f"Agent {agent_path} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        log_error(f"Error running {agent_path}: {e}\n")


if __name__ == "__main__":
    log_info("Starting Genesis AI Pipeline...")

    agents = [
        "agents/paper_reader_agent.py",
        "agents/model_designer_agent.py",
        "agents/experiment_runner_agent.py",
        "agents/report_generator_agent.py",
        "agents/self_mutator_agent.py"  # ✅ New Agent added here
    ]

    for agent in agents:
        run_agent(agent)
    log_info("Genesis AI Pipeline completed.")
