from pinecone import Pinecone

class VectorStore:
    def __init__(self, api_key, index_name):
        self.api_key = api_key
        self.index_name = index_name
        self.pc = None
        self.index = None

    def connect(self):
        """Connect to Pinecone and initialize index."""
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)
        print(f"✅ Connected to Pinecone index: {self.index_name}")

    def upsert_vectors(self, vectors, ids):
        """Upsert vectors into Pinecone index."""
        vectors_list = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        records = list(zip(ids, vectors_list))
        self.index.upsert(vectors=records)
        print(f"✅ Upserted {len(records)} vectors.")

    def query_vector(self, vector, top_k=3):
        """Query Pinecone index with a vector."""
        vector_list = vector.tolist() if hasattr(vector, "tolist") else vector
        result = self.index.query(vector=vector_list, top_k=top_k)
        return result
