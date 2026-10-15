import json
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

from rag_bench.evaluate_retriever import evaluate_retriever

def save_experiment(
    config: dict,
    evaluation: dict,
    output_dir: str
) -> str:
    experiment_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    path_file = output / f"{experiment_id}.json"

    results = {
        "experiment_id": experiment_id,
          "created_at": created_at,
          "config": config,
          "evaluation": evaluation
    }

    with path_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return str(path_file)
