from aether.src.agents.monitoring_agent import MonitoringAgent
from aether.src.agents.anomaly_agent import AnomalyDetectionAgent
from aether.src.agents.optimization_agent import OptimizationAgent
from aether.src.agents.rca_agent import RCAAgent
from aether.src.services.rag_pipeline import RAGPipeline

class AgentOrchestrator:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline
        self.monitor = MonitoringAgent(rag_pipeline)
        self.anomaly = AnomalyDetectionAgent()
        self.rca = RCAAgent(rag_pipeline)
        self.optimizer = OptimizationAgent()

    def run(self):
        print("\n🚀 Starting AETHER-AI Agent Orchestration Pipeline...\n")
        logs = self.monitor.run()
        print("\n🔍 Running Anomaly Detection...\n")
        anomalies = self.anomaly.analyze_logs(logs)
        print("⚠️ Anomalies Found:\n", anomalies)
        print("\n📚 Running RCA Agent (RAG Query)...\n")
        query = "Root cause of recent cost and scaling anomalies?"
        rca_result = self.rca.diagnose(query)
        print("🧠 RCA Result:\n", rca_result)
        print("\n📊 Optimizing Cloud Configuration...\n")
        self.optimizer.train(steps=1000)
        recommendation = self.optimizer.recommend()
        print("📈 Optimization Recommendation:\n", recommendation)
        return {
            "logs": logs,
            "anomalies": anomalies,
            "rca": rca_result,
            "recommendation": recommendation
        }
