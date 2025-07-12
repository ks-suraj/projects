from crewai import Crew, Process, Task
from aether.src.agents.monitoring_agent import MonitoringAgent
from aether.src.agents.anomaly_agent import AnomalyDetectionAgent
from aether.src.agents.rca_agent import RCAAgent
from aether.src.agents.optimization_agent import OptimizationAgent
from aether.src.services.rag_pipeline import RAGPipeline
from aether.config.secrets import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_HOST

def run_crewai_flow():
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

    monitoring_task = Task(
        description="Generate and ingest synthetic cloud logs.",
        agent=monitor,
        expected_output="List of ingested logs."
    )
    anomaly_task = Task(
        description="Detect anomalies in the ingested logs.",
        agent=anomaly,
        expected_output="Summary of detected anomalies."
    )
    rca_task = Task(
        description="Perform root cause analysis on detected anomalies.",
        agent=rca,
        expected_output="Root cause explanation."
    )
    optimization_task = Task(
        description="Optimize cloud resources using RL.",
        agent=optimizer,
        expected_output="Optimization recommendations."
    )

    crew = Crew(
        agents=[monitor, anomaly, rca, optimizer],
        tasks=[monitoring_task, anomaly_task, rca_task, optimization_task],
        process=Process.sequential
    )
    result = crew.kickoff()
    return {
        "logs": result.tasks[0].output,
        "anomalies": result.tasks[1].output,
        "rca": result.tasks[2].output,
        "recommendation": result.tasks[3].output
    }
