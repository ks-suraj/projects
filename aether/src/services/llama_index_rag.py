from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
import faiss
import numpy as np

class LlamaIndexRAG:
    def __init__(self, vector_dim=384):
        self.vector_dim = vector_dim
        Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        Settings.node_parser = SentenceSplitter(chunk_size=512)
        self.faiss_index = faiss.IndexFlatL2(self.vector_dim)
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(
            documents=[],
            storage_context=self.storage_context
        )

    def ingest_docs(self, texts: list[str], metadata: list[dict] = None):
        docs = [Document(text=t, metadata=m or {}) for t, m in zip(texts, metadata or [{}] * len(texts))]
        self.index = VectorStoreIndex.from_documents(
            documents=docs,
            storage_context=self.storage_context
        )
        print(f"✅ Ingested {len(texts)} documents into LlamaIndex (FAISS).")

    def query(self, question: str, top_k=3):
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(question)
        return [{"text": n.text, "score": n.score, "metadata": n.metadata} for n in nodes]
