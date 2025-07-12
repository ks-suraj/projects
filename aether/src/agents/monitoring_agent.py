import uuid
from aether.src.services.rag_pipeline import RAGPipeline
from aether.src.utils.data_generator import generate_synthetic_logs

class MonitoringAgent:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline

    def run(self, count=10):
        print("🟢 MonitoringAgent started...")
        try:
            logs, timestamps = generate_synthetic_logs(count)
            if not logs:
                raise ValueError("No logs generated")
            ids = [str(uuid.uuid4()) for _ in logs]
            self.rag.ingest_docs(logs, ids)
            print(f"✅ MonitoringAgent ingested {len(logs)} logs.")
            return [{"log": log, "timestamp": ts, "id": id_} for log, ts, id_ in zip(logs, timestamps, ids)]
        except Exception as e:
            print(f"❌ MonitoringAgent failed: {str(e)}")
            return [{"log": log, "timestamp": ts, "id": str(uuid.uuid4())} for log, ts in zip(logs, timestamps)] if 'logs' in locals() else []
