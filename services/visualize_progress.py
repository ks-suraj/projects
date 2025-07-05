import os
import sys
import json
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

# Ensure proper import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def visualize_experiment_results():
    console = Console()

    log_path = "experiments/logs/generation_log.json"
    if not os.path.exists(log_path):
        console.print("[bold red]No generation log found! Run the pipeline first.[/bold red]")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        generations = json.load(f)

    if not generations:
        console.print("[bold yellow]No data found in generation log.[/bold yellow]")
        return

    # ✅ 1. Rich Table
    table = Table(title="📊 Experiment Generation History")
    table.add_column("Generation", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Loss", style="red")
    table.add_column("Hyperparameters", style="magenta")

    for gen in generations:
        hyperparams = "\n".join([f"{k}: {v}" for k, v in gen["hyperparameters"].items()])
        table.add_row(
            str(gen["generation"]),
            f"{gen['accuracy']:.2f}",
            f"{gen['loss']:.2f}",
            hyperparams
        )

    console.print(table)

    # ✅ 2. Line Graph (Matplotlib)
    generations_num = [gen["generation"] for gen in generations]
    accuracies = [gen["accuracy"] for gen in generations]
    losses = [gen["loss"] for gen in generations]

    plt.figure(figsize=(8, 5))
    plt.plot(generations_num, accuracies, marker='o', label='Accuracy', color='green')
    plt.plot(generations_num, losses, marker='x', label='Loss', color='red')
    plt.title("Experiment Metric Progress Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("visualizations", exist_ok=True)
    plot_path = "visualizations/experiment_progress.png"
    plt.savefig(plot_path)
    plt.close()

    console.print(f"[bold green]Saved line graph to {plot_path}[/bold green]")

# ✅ Testing friendly
if __name__ == "__main__":
    visualize_experiment_results()
