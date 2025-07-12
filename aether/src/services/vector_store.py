from typing import List
import numpy as np
try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None

class VectorStore:
    def __init__(self, backend="pinecone", vector_dim=384, **kwargs):
        self.backend = backend
        self.vector_dim = vector_dim
        self.config = kwargs
        self.index = None
        self.client = None

    def connect(self):
        if self.backend == "pinecone":
            self._connect_pinecone()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _connect_pinecone(self):
        if Pinecone is None:
            raise ImportError("Pinecone not installed")
        pc = Pinecone(api_key=self.config["api_key"])
        self.index = pc.Index(self.config["index_name"], host=self.config["host"])
        print(f"✅ Connected to Pinecone index: {self.config['index_name']}")

    def upsert_vectors(self, vectors: List, ids: List[str], texts: List[str] = None, batch_size=100):
        vectors = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        texts = texts or [""] * len(ids)
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            if self.backend == "pinecone":
                records = [{"id": id_, "values": vec, "metadata": {"text": txt}} for id_, vec, txt in zip(batch_ids, batch_vectors, batch_texts)]
                self.index.upsert(vectors=records)
                print(f"✅ Upserted {len(batch_vectors)} vectors to Pinecone")
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")

    def query_vector(self, vector, top_k=3):
        if self.backend == "pinecone":
            try:
                result = self.index.query(vector=vector.tolist(), top_k=top_k, include_metadata=True)
                print(f"📊 Pinecone query result: {result}")
                return [(match["id"], match["score"], match.get("metadata", {}).get("text", "")) for match in result["matches"]]
            except Exception as e:
                print(f"❌ Pinecone query failed: {str(e)}")
                return []
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
