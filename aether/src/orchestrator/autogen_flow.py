from autogen import AssistantAgent, GroupChat, GroupChatManager
from aether.src.agents.monitoring_agent import MonitoringAgent
from aether.src.agents.anomaly_agent import AnomalyDetectionAgent
from aether.src.agents.rca_agent import RCAAgent
from aether.src.agents.optimization_agent import OptimizationAgent
from aether.src.services.rag_pipeline import RAGPipeline
from aether.config.secrets import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_HOST

def run_autogen_flow():
    rag_pipeline = RAGPipeline(
        backend="pinecone",
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX_NAME,
        host=PINECONE_HOST,
        vector_dim=384
    )
    monitor = MonitoringAgent(rag_pipeline)
    anomaly = AnomalyDetectionAgent()
    rca = RCAAgent(rag_pipeline)
    optimizer = OptimizationAgent()

    logs = monitor.run()
    anomalies = anomaly.analyze_logs(logs)
    rca_result = rca.diagnose("Root cause of recent cost and scaling anomalies?")
    optimizer.train(steps=1000)
    recommendation = optimizer.recommend()

    return {
        "logs": logs,
        "anomalies": anomalies,
        "rca": rca_result,
        "recommendation": recommendation
    }
