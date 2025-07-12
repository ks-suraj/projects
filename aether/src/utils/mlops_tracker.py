import mlflow
import wandb

def log_to_mlflow(experiment_name: str, metrics: dict, params: dict):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

def log_to_wandb(project_name: str, metrics: dict, params: dict):
    wandb.init(project=project_name, config=params, anonymous="allow")
    wandb.log(metrics)
    wandb.finish()
