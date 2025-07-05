import json
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rich.console import Console
from rich.table import Table

def visualize_experiment_results():
    console = Console()

    try:
        # Load generation log
        with open("experiments/logs/generation_log.json", "r") as f:
            generations = json.load(f)

        if not generations:
            console.print("[bold red]No generations found.[/bold red]")
            return

        console.print("[bold cyan]📊 Generations Summary Table:[/bold cyan]")
        table = Table(title="Experiment Generations Summary")
        table.add_column("Gen", style="cyan", no_wrap=True)
        table.add_column("Accuracy (%)", style="magenta")
        table.add_column("Loss", style="green")

        accuracies = []
        losses = []
        generations_list = []

        for gen in generations:
            gen_num = str(gen["generation"])
            acc = gen["accuracy"]
            loss = gen["loss"]
            table.add_row(gen_num, str(acc), str(loss))
            generations_list.append(gen_num)
            accuracies.append(acc)
            losses.append(loss)

        console.print(table)

        # ✅ Plot Accuracy & Loss over Generations
        os.makedirs("visualizations", exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(generations_list, accuracies, marker='o', label="Accuracy (%)", color="blue")
        plt.plot(generations_list, losses, marker='x', label="Loss", color="red")
        plt.title("Metrics Across Generations")
        plt.xlabel("Generation")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("visualizations/generation_metrics.png")
        plt.close()

        console.print("[bold green]Saved graph to visualizations/generation_metrics.png[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Visualization Error: {e}[/bold red]")

if __name__ == "__main__":
    visualize_experiment_results()
