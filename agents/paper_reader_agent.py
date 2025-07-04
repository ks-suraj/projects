import sys
import os

# Ensure proper import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.api_client import query_openrouter
from utils.logger import log_info, log_error

def read_paper_and_summarize(paper_content):
    prompt = f"Summarize this research paper:\n\n{paper_content}"
    try:
        log_info("Starting paper summarization...")
        summary = query_openrouter(prompt)
        log_info("Paper summarization completed successfully.")
        return summary
    except Exception as e:
        log_error(f"Paper summarization failed: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    paper = "This paper introduces a novel method for optimizing deep learning architectures using evolutionary strategies."
    summary = read_paper_and_summarize(paper)
    if summary:
        print("📄 Paper Summary:\n", summary)
