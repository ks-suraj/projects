import scrapy
from scrapy.crawler import CrawlerProcess
import requests
import io
from PyPDF2 import PdfReader
try:
    from PyPDF2.errors import PdfReadError
except ImportError:
    PdfReadError = Exception  # Fallback for older/newer PyPDF2 versions

class ArxivMinimalSpider(scrapy.Spider):
    name = "arxiv_minimal_spider"
    start_urls = ["https://arxiv.org/list/cs.AI/recent"]

    def parse(self, response):
        # Extract the first PDF link (latest paper)
        pdf_link = response.xpath("//a[@title='Download PDF']/@href").get()
        if pdf_link:
            pdf_url = f"https://arxiv.org{pdf_link}"
            try:
                # Fetch PDF using the provided mechanism
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()
                pdf_bytes = response.content
                reader = PdfReader(io.BytesIO(pdf_bytes))
                full_text = "\n".join([page.extract_text() or '' for page in reader.pages])

                # Save description to file
                with open("data/paper_input.txt", 'w', encoding="utf-8") as txt_file:
                    txt_file.write(full_text)
            except (PdfReadError, requests.RequestException) as e:
                print(f"Error processing {pdf_url}: {str(e)}")
        else:
            print("No PDF link found on page")

if __name__ == "__main__":
    process = CrawlerProcess({
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "DOWNLOAD_DELAY": 0.5,
    })
    process.crawl(ArxivMinimalSpider)
    process.start()