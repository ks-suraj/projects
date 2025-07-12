import yaml

def generate_k8s_yaml(service_name: str, replicas: int, cpu_threshold: float):
    k8s_config = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": f"{service_name}-hpa"},
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": service_name
            },
            "minReplicas": 1,
            "maxReplicas": replicas,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": int(cpu_threshold * 100)
                        }
                    }
                }
            ]
        }
    }
    with open(f"aether/data/processed/{service_name}_hpa.yaml", "w") as f:
        yaml.dump(k8s_config, f)
    return yaml.dump(k8s_config)
