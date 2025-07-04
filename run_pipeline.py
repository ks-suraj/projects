import subprocess
import sys
from utils.logger import log_info, log_error

def run_agent(agent_path):
    try:
        log_info(f"Running {agent_path}...")
        result = subprocess.run(
            [sys.executable, agent_path],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        log_info(f"Agent {agent_path} completed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        log_error(f"Error running {agent_path}: {e.stderr}")

def run_pipeline():
    agents = [
        "agents/paper_reader_agent.py",
        "agents/model_designer_agent.py",
        "agents/experiment_runner_agent.py"
    ]
    for agent in agents:
        run_agent(agent)

if __name__ == "__main__":
    log_info("Starting Genesis AI Pipeline...")
    run_pipeline()
    log_info("Genesis AI Pipeline completed.")
