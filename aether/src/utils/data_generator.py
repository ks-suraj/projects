import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_logs(count=100):
    clouds = ["AWS", "GCP", "Azure"]
    services = ["S3", "EC2", "BigQuery", "VM", "Blob"]
    metrics = ["cost", "cpu_usage", "requests"]
    
    data = {
        "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(count)],
        "cloud": np.random.choice(clouds, count),
        "service": np.random.choice(services, count),
        "metric": np.random.choice(metrics, count),
        "value": np.random.uniform(0, 100, count),
        "description": [f"{cloud} {service} {metric} anomaly" for cloud, service, metric in zip(
            np.random.choice(clouds, count),
            np.random.choice(services, count),
            np.random.choice(metrics, count)
        )]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("aether/data/raw/synthetic_logs.csv", index=False)
    return df["description"].tolist(), df["timestamp"].astype(str).tolist()
