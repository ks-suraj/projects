import os
import sys
import requests
import xml.etree.ElementTree as ET
import time
import json
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_info, log_error
from utils.api_client import call_openrouter_api

def fetch_latest_arxiv_papers(query="Deep Learning", max_results=1, max_retries=3):
    """Fetches latest papers from arXiv via API (ATOM Feed)."""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    for attempt in range(max_retries):
        try:
            log_info(f"Fetching latest {max_results} papers from arXiv (attempt {attempt + 1}/{max_retries})")
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            papers = []
            root = ET.fromstring(response.text)
            
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                arxiv_id = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
                paper = {
                    "title": entry.find("{http://www.w3.org/2005/Atom}title").text.strip(),
                    "summary": entry.find("{http://www.w3.org/2005/Atom}summary").text.strip(),
                    "authors": [author.find("{http://www.w3.org/2005/Atom}name").text.strip() 
                               for author in entry.findall(".//{http://www.w3.org/2005/Atom}author")],
                    "published": entry.find("{http://www.w3.org/2005/Atom}published").text,
                    "updated": entry.find("{http://www.w3.org/2005/Atom}updated").text,
                    "arxiv_id": arxiv_id,
                    "categories": [cat.get("term") for cat in entry.findall(".//{http://www.w3.org/2005/Atom}category")],
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                }
                papers.append(paper)
            
            log_info(f"Successfully fetched {len(papers)} papers.")
            return papers

        except requests.exceptions.RequestException as e:
            log_error(f"Request failed (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
        except Exception as e:
            log_error(f"Unexpected error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

    log_error("All attempts to fetch papers failed")
    return []

def summarize_papers(papers, max_retries=3):
    """Summarizes papers using OpenRouter API with retry mechanism."""
    summarized_papers = []
    
    for paper in papers:
        for attempt in range(max_retries):
            try:
                prompt = f"""Analyze this research paper and provide a structured summary:

Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Categories: {', '.join(paper['categories'])}
Published: {paper['published']}
Abstract: {paper['summary']}

Please provide:
1. Main research contribution
2. Key methodology/approach
3. Primary findings/results
4. Potential applications
"""
                messages = [{"role": "user", "content": prompt}]
                summary = call_openrouter_api(messages)
                
                if not summary or len(summary.strip()) < 50:
                    raise ValueError("Invalid or too short summary received")
                    
                paper['ai_summary'] = summary
                summarized_papers.append(paper)
                log_info(f"AI Summary completed for: {paper['title']}")
                break
                
            except Exception as e:
                log_error(f"Summary attempt {attempt + 1} failed for '{paper['title']}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    paper['ai_summary'] = None
                    summarized_papers.append(paper)

    return summarized_papers

def save_summaries_and_metadata(papers):
    """Saves all paper summaries and metadata, and also saves each summary as a separate file for the pipeline."""
    try:
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all papers with summaries
        all_data_path = f"data/papers_{timestamp}.json"
        with open(all_data_path, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        log_info(f"Saved all paper data to {all_data_path}")

        # Save each paper summary as a separate file for pipeline use
        for idx, paper in enumerate(papers):
            if paper.get('ai_summary'):
                pipeline_path = f"data/paper_input_{idx+1}_{timestamp}.txt"
                with open(pipeline_path, "w", encoding="utf-8") as f:
                    summary_data = {
                        "title": paper['title'],
                        "authors": paper['authors'],
                        "categories": paper['categories'],
                        "summary": paper['ai_summary'],
                        "arxiv_id": paper['arxiv_id'],
                        "pdf_url": paper['pdf_url']
                    }
                    json.dump(summary_data, f, indent=2, ensure_ascii=False)
                log_info(f"Saved pipeline input to {pipeline_path}")
            else:
                log_error(f"No valid summary for paper: {paper.get('title', 'Unknown Title')}")
                
    except Exception as e:
        log_error(f"Error saving paper data: {e}")

def main():
    log_info("Starting arXiv Scraper Pipeline...")
    
    try:
        papers = fetch_latest_arxiv_papers(
            query="Deep Learning",
            max_results=1
        )
        
        if papers:
            summarized_papers = summarize_papers(papers)
            save_summaries_and_metadata(summarized_papers)
            log_info("arXiv Scraper Pipeline completed successfully.")
            log_info("You can now run the Paper Reader Agent to process the saved paper summaries.")
        else:
            log_error("No papers were fetched. Pipeline cannot continue.")
            
    except Exception as e:
        log_error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()  
# This code is a complete arXiv scraper pipeline that fetches the latest research papers,