from aether.src.services.vector_store import VectorStore
from aether.src.services.embedding import EmbeddingEngine
from typing import List, Union
import numpy as np

class RAGPipeline:
    def __init__(self, backend="pinecone", vector_dim=384, **kwargs):
        self.backend = backend
        self.vector_dim = vector_dim
        self.config = kwargs
        self.embedder = EmbeddingEngine("all-MiniLM-L6-v2")
        self.vector_store = VectorStore(backend=backend, vector_dim=vector_dim, **kwargs)
        self.vector_store.connect()

    def ingest_docs(self, texts: List[str], ids: List[str]):
        if not texts or not ids:
            print("⚠️ No texts or IDs provided for ingestion.")
            return
        try:
            vectors = self.embedder.embed(texts)
            if vectors.size == 0:
                print("⚠️ No valid embeddings generated.")
                return
            self.vector_store.upsert_vectors(vectors, ids, texts)
            print(f"✅ Ingested {len(texts)} documents.")
        except Exception as e:
            print(f"❌ RAG ingestion failed: {str(e)}")
            raise

    def query(self, text: Union[str, List[str]], top_k=3):
        try:
            query_vec = self.embedder.embed(text)
            if query_vec.size == 0:
                print("⚠️ Invalid query embedding.")
                return []
            results = self.vector_store.query_vector(query_vec[0] if query_vec.ndim > 1 else query_vec, top_k=top_k)
            print(f"📄 Retrieved documents: {[(id_, score, text) for id_, score, text in results]}")
            return results
        except Exception as e:
            print(f"❌ RAG query failed: {str(e)}")
            return []
