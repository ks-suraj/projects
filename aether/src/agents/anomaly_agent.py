from aether.src.services.llm_client import LLMClient
from typing import List

class AnomalyDetectionAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, logs: List[dict]) -> str:
        if not logs:
            return "No logs provided for anomaly detection."
        log_texts = [log["log"] for log in logs]
        prompt = f"Analyze the following logs and identify potential anomalies or risks:\n\n{log_texts}\n\nProvide a bullet-point summary."
        try:
            response = self.llm.generate(prompt)
            return response
        except Exception as e:
            print(f"❌ Anomaly detection failed: {str(e)}")
            return f"Anomaly detection failed: {str(e)}"
