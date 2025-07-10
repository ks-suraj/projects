
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(text):
    """Basic cleaning (can be extended later)."""
    cleaned_text = text.replace("\n", " ").replace("\t", " ").strip()
    return cleaned_text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Chunk the cleaned text into smaller parts for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks
