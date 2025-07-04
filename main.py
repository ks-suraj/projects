from services.arxiv_scraper import fetch_latest_arxiv_papers, summarize_papers
from utils.logger import log_info

if __name__ == "__main__":
    log_info("🚀 Starting Genesis AI Research Pipeline...")

    papers = fetch_latest_arxiv_papers(query="machine learning", max_results=3)
    if papers:
        summarized_papers = summarize_papers(papers)

        for idx, paper in enumerate(summarized_papers, start=1):
            print(f"\n📄 Paper {idx}: {paper['title']}\n")
            print("AI-Generated Summary:\n", paper['ai_summary'])
            print("=" * 80)

    else:
        log_info("❌ No papers fetched.")

    log_info("✅ Pipeline execution completed successfully.")
