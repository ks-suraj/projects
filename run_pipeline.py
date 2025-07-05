import subprocess
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_info, log_error
from services.visualize_progress import visualize_experiment_results  # ✅ Visualization Import

def run_agent(agent_path):
    try:
        log_info(f"Running {agent_path}...")
        subprocess.run([sys.executable, agent_path], check=True)
        log_info(f"Agent {agent_path} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        log_error(f"Error running {agent_path}: {e}\n")

if __name__ == "__main__":
    log_info("🚀 Starting Genesis AI Pipeline...")

    agents = [
        "services/arxiv_scraper.py",
        "agents/paper_reader_agent.py",
        "agents/model_designer_agent.py",
        "agents/experiment_runner_agent.py",
        "agents/report_generator_agent.py",
        "agents/self_mutator_agent.py"
    ]

    for agent in agents:
        run_agent(agent)

    # ✅ Visualization Step
    try:
        log_info("📊 Running Visualization Step...")
        visualize_experiment_results()
        log_info("✅ Visualization completed successfully.\n")
    except Exception as e:
        log_error(f"Visualization Step Error: {e}\n")

    log_info("🎉 Genesis AI Pipeline completed.")
    log_info("All agents executed successfully. Check reports, results & visualizations.")
