from utils.logger import log_info, log_error
from utils.api_client import query_openrouter

import requests
import xml.etree.ElementTree as ET


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
    """Summarizes papers using the OpenRouter API."""
    for paper in papers:
        try:
            prompt = f"Summarize this paper:\n\nTitle: {paper['title']}\n\nAbstract: {paper['summary']}"
            summary = query_openrouter(prompt)
            paper['ai_summary'] = summary
            log_info(f"AI Summary completed for: {paper['title']}")
        except Exception as e:
            log_error(f"Failed to summarize paper '{paper['title']}': {e}")
            paper['ai_summary'] = None
    return papers


if __name__ == "__main__":
    log_info("Starting arXiv Scraper Pipeline...")

    papers = fetch_latest_arxiv_papers(query="machine learning", max_results=3)
    if papers:
        summarized_papers = summarize_papers(papers)

        for idx, paper in enumerate(summarized_papers, start=1):
            print(f"\n📄 Paper {idx}: {paper['title']}\n")
            print("AI-Generated Summary:\n", paper['ai_summary'])
            print("=" * 80)

    log_info("Pipeline completed successfully.")
