import pytest
from rag_bench.evaluate_retriever import evaluate_retriever
from rag_bench.evaluate_retriever import compare_queries

@pytest.mark.parametrize("dataset, retriever_results, k, expected, should_fail",[
    (
        {
            "queries": [
                {"id": "q1", "relevant": ["A", "C"]},
                {"id": "q2", "relevant": ["E"]}
            ]
        },
        {
            "results": {
                "q1": ["A", "B", "C"],
                "q2": ["D", "E", "F"]
            }
        },
        2,
        {
            "queries_evaluated": 2,
            "recall_at_k": 0.75,
            "precision_at_k": 0.5,
            "mrr_at_k": 0.75,
            "details": [
                {
                    "query_id": "q1",
                    "recall": 0.5,
                    "precision": 0.5,
                    "reciprocal_rank": 1.0
                },
                {
                    "query_id": "q2",
                    "recall": 1.0,
                    "precision": 0.5,
                    "reciprocal_rank": 0.5
                }
            ]
        },
        False
    ),
    (
        {
            "queries": [
                {"id": "q1", "relevant": ["A", "C"]},
                {"id": "q2", "relevant": ["E"]}
            ]
        },
        {
            "results": {
                "q1": ["A", "B", "C"]
            }
        },
        2,
        ValueError,
        True
    ),
    (
        {
            "queries": [
                {"id": "q1", "relevant": ["A", "C"]},
                {"id": "q2", "relevant": ["E"]}
            ]
        },
        {
            "results": {
                "q1": [],
                "q2": []
            }
        },
        2,
        {
            "queries_evaluated": 2,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr_at_k": 0.0,
            "details": [
                {
                    "query_id": "q1",
                    "recall": 0.0,
                    "precision": 0.0,
                    "reciprocal_rank": 0.0
                },
                {
                    "query_id": "q2",
                    "recall": 0.0,
                    "precision": 0.0,
                    "reciprocal_rank": 0.0
                }
            ]
        },
        False
    ),
    (
        {
            "queries": [
                {"id": "q1", "relevant": ["A", "C"]},
                {"id": "q2", "relevant": ["E"]}
            ]
        },
        {
            "results": {
            }
        },
        2,
        ValueError,
        True
    ),
])

def test_evaluate_retriever(dataset, retriever_results, k, expected, should_fail):
    if should_fail:
        with pytest.raises(ValueError):
            evaluate_retriever(dataset, retriever_results, k)
    else:
        assert(evaluate_retriever(dataset, retriever_results, k) == expected)

@pytest.mark.parametrize("results_a, results_b, expected, should_fail", [
    (
        {
            "queries_evaluated": 2,
            "recall_at_k": 0.75,
            "precision_at_k": 0.5,
            "mrr_at_k": 0.75,
            "details": [
                {
                    "query_id": "q1",
                    "recall": 0.5,
                    "precision": 0.5,
                    "reciprocal_rank": 1.0
                },
                {
                    "query_id": "q2",
                    "recall": 1.0,
                    "precision": 0.5,
                    "reciprocal_rank": 0.5
                }
            ]
        },
        {
            "queries_evaluated": 2,
            "recall_at_k": 0.75,
            "precision_at_k": 0.5,
            "mrr_at_k": 0.75,
            "details": [
                {
                    "query_id": "q2",
                    "recall": 0.5,
                    "precision": 0.5,
                    "reciprocal_rank": 1.0
                },
                {
                    "query_id": "q1",
                    "recall": 1.0,
                    "precision": 0.5,
                    "reciprocal_rank": 0.5
                }
            ]
        },
        [
            {
                "query_id": "q1",
                "recall_a": 0.5,
                "recall_b": 1.0,
                "delta": 0.5
            },
            {
                "query_id": "q2",
                "recall_a": 1.0,
                "recall_b": 0.5,
                "delta": -0.5
            }
        ],
        False
    ),
    (
        {
            "queries_evaluated": 2,
            "recall_at_k": 0.75,
            "precision_at_k": 0.5,
            "mrr_at_k": 0.75,
            "details": [
                {
                    "query_id": "q1",
                    "recall": 0.5,
                    "precision": 0.5,
                    "reciprocal_rank": 1.0
                },
                {
                    "query_id": "q2",
                    "recall": 1.0,
                    "precision": 0.5,
                    "reciprocal_rank": 0.5
                }
            ]
        },
        {
            "queries_evaluated": 2,
            "recall_at_k": 0.75,
            "precision_at_k": 0.5,
            "mrr_at_k": 0.75,
            "details": [
                {
                    "query_id": "q3",
                    "recall": 0.5,
                    "precision": 0.5,
                    "reciprocal_rank": 1.0
                },
                {
                    "query_id": "q1",
                    "recall": 1.0,
                    "precision": 0.5,
                    "reciprocal_rank": 0.5
                }
            ]
        },
        ValueError,
        True
    ),
])


def test_compare_queries(results_a, results_b, expected, should_fail):
    if should_fail:
        with pytest.raises(ValueError):
            compare_queries(results_a, results_b)
    else:
        assert(compare_queries(results_a, results_b) == expected)
