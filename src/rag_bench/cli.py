import json
from rag_bench.evaluate_retriever import evaluate_retriever
import typer
from pathlib import Path

app = typer.Typer()

def main():
    app()

@app.callback()
def callback():
    """RAG retrieval benchmarking CLI."""

@app.command()
def evaluate(
    dataset: Path = typer.Option(..., help="Evaluation dataset"),
    results: Path = typer.Option(..., help="Retriever results"),
    k: int = typer.Option(5, help="Number of retrieved chunks")
):
    try:
        if not dataset.exists():
            raise typer.BadParameter("Dataset file not found")

        if not results.exists():
            raise typer.BadParameter("Results file not found")

        if k <= 0:
            raise typer.BadParameter("K must be greater than 0")

        with dataset.open("r", encoding="utf-8") as file:
            dataset_json = json.load(file)

        with results.open("r", encoding="utf-8") as file:
            results_json = json.load(file)

        metric = evaluate_retriever(dataset_json, results_json, k)
    except json.JSONDecodeError:
        typer.error("Error: Invalid JSON file", err=True)
        raise typer.Exit(code=1)
    except ValueError:
        typer.echo("Error: Invalid Dataset", err=True)
        raise typer.Exit(code=1)
    except OSError:
        typer.echo("Error: File unreadable", err=True)
        raise typer.Exit(code=1)

    recall = metric["recall_at_k"]
    precision = metric["precision_at_k"]
    mrr = metric["mrr_at_k"]

    print("RAG Benchmark")
    print("-----------------------------")
    print("Queries evaluated : ", metric["queries_evaluated"])
    print("")
    print(f"Recall@{k}      : {recall:.3f}")
    print(f"Precision@{k}   : {precision:.3f}")
    print(f"MRR@{k}         : {mrr:.3f}")

if __name__ == "__main__":
    main()
