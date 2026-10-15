import json

from rag_bench.metrics.ranking import reciprocal_rank_at_k
from rag_bench.metrics.recall import recall_at_k
from rag_bench.metrics.precision import precision_at_k
from rag_bench.validation import validate_dataset

def evaluate_retriever(
    dataset: dict,
    retriever_results: dict,
    k: int
) -> dict:
    details = []
    recall_all = 0.0
    precision_all = 0.0
    queries = 0

    validate_dataset(dataset)

    if not dataset["queries"]:
       raise ValueError("Dataset must contain at least one query")


    for query in dataset["queries"]:
        query_id = query["id"]
        relevant = query["relevant"]
        results = retriever_results["results"]

        if query_id not in results:
            raise ValueError(f"Missing retrieval results for query: {query_id}")

        retriever = results[query_id]

        recall = recall_at_k(retriever, relevant, k)
        precision = precision_at_k(retriever, relevant, k)
        rr = reciprocal_rank_at_k(retriever, relevant, k)

        queries += 1
        recall_all += recall
        precision_all += precision
        details.append({
            "query_id": query_id,
            "recall": recall,
            "precision": precision,
            "reciprocal_rank": rr
        })

    mrr = sum(
        detail["reciprocal_rank"]
        for detail in details
    ) / len(details)

    return {
        "queries_evaluated": queries,
        "recall_at_k": recall_all / queries,
        "precision_at_k": precision_all / queries,
        "mrr_at_k": mrr,
        "details": details
    }

def compare_queries(
    results_a: dict,
    results_b: dict
) -> list[dict]:
    compare = []

    details_by_id = {
        detail["query_id"]: detail
        for detail in results_b["details"]
    }

    ids_a = {
        detail["query_id"]
        for detail in results_a["details"]
    }
    ids_b = set(details_by_id)
    if ids_a != ids_b:
        raise ValueError(
            f"Query IDs mismatch: {ids_a ^ ids_b}"
        )

    for detail_a in results_a["details"]:
        recall_a = detail_a["recall"]
        detail_b = details_by_id[detail_a["query_id"]]
        recall_b = detail_b["recall"]

        compare.append({
            "query_id": detail_a["query_id"],
            "recall_a": recall_a,
            "recall_b": recall_b,
            "delta": recall_b - recall_a
        })

    return compare
