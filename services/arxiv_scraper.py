import os
import sys
import requests
import xml.etree.ElementTree as ET

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_info, log_error
from utils.api_client import call_openrouter_api

def fetch_latest_arxiv_papers(query="machine learning", max_results=3):
    """Fetches latest papers from arXiv via RSS (ATOM Feed)."""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    log_info(f"Fetching latest {max_results} papers from arXiv for query: '{query}'")
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
    except Exception as e:
        log_error(f"Failed to fetch arXiv papers: {e}")
        return []

    papers = []
    try:
        root = ET.fromstring(response.text)
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            papers.append({"title": title, "summary": summary})
    except Exception as e:
        log_error(f"Error parsing arXiv feed: {e}")
        return []

    log_info(f"Successfully fetched {len(papers)} papers.")
    return papers

def summarize_papers(papers):
    """Summarizes papers using OpenRouter API."""
    for paper in papers:
        try:
            prompt = f"Summarize this paper:\n\nTitle: {paper['title']}\n\nAbstract: {paper['summary']}"
            messages = [{"role": "user", "content": prompt}]
            summary = call_openrouter_api(messages)
            paper['ai_summary'] = summary
            log_info(f"AI Summary completed for: {paper['title']}")
        except Exception as e:
            log_error(f"Failed to summarize paper '{paper['title']}': {e}")
            paper['ai_summary'] = None
    return papers

def save_summary_to_file(papers):
    """Saves the summary of first paper to paper_input.txt (for pipeline input)."""
    os.makedirs("data", exist_ok=True)
    file_path = "data/paper_input.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        # Save only the first paper summary for pipeline
        f.write(papers[0]['ai_summary'] if papers[0]['ai_summary'] else "")
    log_info(f"Saved paper summary to {file_path}")

if __name__ == "__main__":
    log_info("Starting arXiv Scraper Pipeline...")

    papers = fetch_latest_arxiv_papers(query="machine learning", max_results=3)
    if papers:
        summarized_papers = summarize_papers(papers)
        save_summary_to_file(summarized_papers)

    log_info("arXiv Scraper Pipeline completed successfully.")
    log_info("You can now run the Paper Reader Agent to process the saved paper summary.")  
    