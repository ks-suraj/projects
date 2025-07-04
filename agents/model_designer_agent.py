import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.logger import log_info, log_error
from utils.api_client import query_openrouter
import json

def design_new_model(paper_summary, save_dir="models/generated_models/"):
    """Uses LLM to generate new neural model design from paper summary."""
    log_info("Starting model design agent...")

    prompt = f"""
    You are an AI Research Assistant.

    Given the following research paper summary, design a NEW deep learning architecture inspired by it.

    Please provide:
    1. A short textual description of the architecture (max 150 words).
    2. A sample PyTorch-like pseudocode (no need for full runnable code).

    Research Paper Summary:
    {paper_summary}
    """

    try:
        response = query_openrouter(prompt)
        log_info("Model design generated successfully.")

        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "generated_model.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"summary": paper_summary, "design": response}, f, indent=2)

        log_info(f"Saved model design to {file_path}")
        return response

    except Exception as e:
        log_error(f"Failed to generate model design: {e}")
        return None


if __name__ == "__main__":
    test_summary = """
    This is a sample paper summary discussing a novel convolutional transformer hybrid model for image classification,
    which combines convolution layers with attention-based modules to improve accuracy and efficiency on the ImageNet benchmark.
    """
    design = design_new_model(test_summary)
    if design:
        print("📝 AI-Generated Model Design:\n", design)
