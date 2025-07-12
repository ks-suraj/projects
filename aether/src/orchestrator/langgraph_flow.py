from langgraph.graph import StateGraph
from aether.src.agents.monitoring_agent import MonitoringAgent
from aether.src.agents.anomaly_agent import AnomalyDetectionAgent
from aether.src.agents.rca_agent import RCAAgent
from aether.src.agents.optimization_agent import OptimizationAgent
from aether.src.services.rag_pipeline import RAGPipeline
from aether.src.services.llm_client import LLMClient
from aether.config.secrets import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_HOST
from typing import TypedDict, Dict, List

class AetherState(TypedDict):
    logs: List[Dict]
    anomalies: str
    rca: str
    recommendation: Dict

def build_aether_graph():
    rag_pipeline = RAGPipeline(
        backend="pinecone",
        vector_dim=384,
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX_NAME,
        host=PINECONE_HOST
    )
    llm_client = LLMClient(model="mistralai/mistral-nemo:free")
    monitor = MonitoringAgent(rag_pipeline)
    anomaly = AnomalyDetectionAgent(llm_client)
    rca = RCAAgent(rag_pipeline, llm_client)
    optimizer = OptimizationAgent()

    def monitor_node(state: AetherState) -> AetherState:
        logs = monitor.run(count=10)
        return {"logs": logs}

    def anomaly_node(state: AetherState) -> AetherState:
        anomalies = anomaly.run(state["logs"])
        return {"anomalies": anomalies}

    def rca_node(state: AetherState) -> AetherState:
        rca_result = rca.run("Root cause of recent cost and scaling anomalies?")
        return {"rca": rca_result}

    def optimize_node(state: AetherState) -> AetherState:
        recommendation = optimizer.run(state["logs"])
        return {"recommendation": recommendation}

    workflow = StateGraph(AetherState)
    workflow.add_node("monitor", monitor_node)
    workflow.add_node("anomaly", anomaly_node)
    workflow.add_node("rca", rca_node)
    workflow.add_node("optimize", optimize_node)

    workflow.set_entry_point("monitor")
    workflow.add_edge("monitor", "anomaly")
    workflow.add_edge("anomaly", "rca")
    workflow.add_edge("rca", "optimize")
    workflow.add_edge("optimize", "__end__")

    return workflow.compile()
