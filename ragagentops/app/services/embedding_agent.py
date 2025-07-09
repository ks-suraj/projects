from sentence_transformers import SentenceTransformer

class EmbeddingAgent:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded.")

    def embed_text(self, text):
        embedding = self.model.encode(text).tolist()
        return embedding
