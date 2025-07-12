from aether.src.services.rag_pipeline import RAGPipeline
from aether.src.services.llm_client import LLMClient

class RCAAgent:
    def __init__(self, rag_pipeline: RAGPipeline, llm_client: LLMClient):
        self.rag = rag_pipeline
        self.llm = llm_client

    def run(self, query: str):
        try:
            retrieved_docs = self.rag.query(query, top_k=3)
            if not retrieved_docs:
                print("⚠️ No documents retrieved for RCA.")
                context = "No relevant documents found."
            else:
                context = "\n".join([doc[2] for doc in retrieved_docs])
                print(f"📄 RCA context: {context}")
            prompt = f"Based on the following logs:\n{context}\n\nProvide a concise root cause analysis for: {query}"
            response = self.llm.generate(prompt)
            return response
        except Exception as e:
            print(f"❌ RCA failed: {str(e)}")
            return f"RCA failed: {str(e)}"
