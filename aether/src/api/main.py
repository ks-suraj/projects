from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from aether.src.agents.agent_orchestrator import AgentOrchestrator
from aether.src.services.rag_pipeline import RAGPipeline
from aether.config.secrets import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_HOST

app = FastAPI(title="AETHER-AI API")

class QueryInput(BaseModel):
    query: str
    top_k: int = 3

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/run_pipeline")
async def run_pipeline():
    try:
        rag_pipeline = RAGPipeline(
            backend="pinecone",
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME,
            host=PINECONE_HOST,
            vector_dim=384
        )
        orchestrator = AgentOrchestrator(rag_pipeline)
        result = orchestrator.run()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_rag")
async def query_rag(input: QueryInput):
    try:
        rag_pipeline = RAGPipeline(
            backend="pinecone",
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME,
            host=PINECONE_HOST,
            vector_dim=384
        )
        result = rag_pipeline.query(input.query, top_k=input.top_k)
        return {"query": input.query, "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
