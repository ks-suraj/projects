
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
    """Loads the Sentence Transformer model."""
    model = SentenceTransformer(model_name, device=device)
    return model

def embed_and_store(chunks, model):
    """Embeds text chunks and stores them in FAISS index."""
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings
