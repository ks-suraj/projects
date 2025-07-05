import os
import sys
import json

# Setup project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import log_info, log_error

def run_report_generator():
    try:
        log_info("📄 Starting Report Generator Agent...")

        # Load Paper Summary
        with open("outputs/paper_summary.txt", "r", encoding="utf-8") as f:
            paper_summary = f.read()

        # Load Model Design
        with open("models/generated_models/generated_model.json", "r", encoding="utf-8") as f:
            model_design = json.load(f)

        # Load Experiment Results
        with open("experiments/results/experiment_result.json", "r", encoding="utf-8") as f:
            experiment_result = json.load(f)

        # Combine into Final Report
        final_report = {
            "Paper Summary": paper_summary,
            "Model Design": model_design,
            "Experiment Result": experiment_result
        }

        # Save Report
        os.makedirs("reports", exist_ok=True)
        with open("reports/final_report.json", "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=4, ensure_ascii=False)

        log_info("✅ Report generated successfully. Check reports/final_report.json")

    except Exception as e:
        log_error(f"Report Generator Agent Error: {str(e)}")


if __name__ == "__main__":
    run_report_generator()
