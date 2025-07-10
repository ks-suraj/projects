
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = " ".join([p.get_text() for p in soup.find_all("p")])
    return text

def load_text(raw_text):
    return raw_text.strip()
