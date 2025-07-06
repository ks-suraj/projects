import os
import sys
# Add project root to sys.path  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from utils.api_client import call_openrouter_api
from utils.logger import log_info, log_error

def load_paper_summary():
    input_path = "outputs/paper_summary.txt"
    if not os.path.exists(input_path):
        log_error(f"Paper summary not found at {input_path}")
        return None
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()

def design_model(paper_summary):
    system_message = {
        "role": "system",
        "content": (
            "You are an expert AI research engineer. "
            "Given the following paper summary, design a deep learning model inspired by it. "
            "Respond ONLY with valid python code with perfect indentation, perfect working, ready to deploy, and correct parameter parsing. "
            "Do not include explanations or extra text."
        )
    }
    user_message = {
        "role": "user",
        "content": f"Paper Summary:\n{paper_summary}"
    }
    messages = [system_message, user_message]
    response = call_openrouter_api(messages)
    print("LLM raw response:", response)  # Debug print
    return response  # Return raw python code

def save_python_file(python_code):
    codes_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "codes")
    os.makedirs(codes_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_{timestamp}.py"
    filepath = os.path.join(codes_dir, filename)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"# Generated on {now_str}\n"
    content += "# Model implementation below\n\n"
    content += python_code
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    log_info(f"Model code saved to {filepath}")

def main():
    log_info("Starting Model Designer Agent...")
    paper_summary = load_paper_summary()
    if not paper_summary:
        log_error("No paper summary to process.")
        return

    python_code = design_model(paper_summary)
    save_python_file(python_code)
    log_info("Model Designer Agent completed successfully.")

if __name__ == "__main__":
    main()