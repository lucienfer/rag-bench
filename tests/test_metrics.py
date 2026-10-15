import pytest
from rag_bench.metrics.precision import precision_at_k
from rag_bench.metrics.ranking import mrr_at_k, reciprocal_rank_at_k
from rag_bench.metrics.recall import recall_at_k


@pytest.mark.parametrize("parameter", [
    {
        "retrieved": ["A", "B", "C"],
        "relevant": ["A", "B", "C"],
        "k": 3,
        "result": 1.0
    },
    {
        "retrieved": ["A", "B", "C"],
        "relevant": ["A", "C"],
        "k": 3,
        "result": 2/3
    },
    {
        "retrieved": ["A", "B", "C"],
        "relevant": ["D"],
        "k": 3,
        "result": 0.0
    },
    {
        "retrieved": ["A", "B"],
        "relevant": ["A"],
        "k": 5,
        "result": 0.2
    }
])

def test_precision_at_k(parameter):
    assert precision_at_k(
        parameter["retrieved"], parameter["relevant"], parameter["k"]
    ) == parameter["result"]

@pytest.mark.parametrize("parameter", [
    {
        "retrieved": ["A", "B", "C"],
        "relevant": ["A", "C"],
        "k": 3,
        "result": 1.0
    },
    {
        "retrieved": ["A", "B", "C"],
        "relevant": ["A", "C"],
        "k": 2,
        "result": 0.5
    },
    {
        "retrieved": ["A", "B"],
        "relevant": ["C"],
        "k": 2,
        "result": 0.0
    },
    {
        "retrieved": ["A", "A", "B"],
        "relevant": ["A", "B"],
        "k": 2,
        "result": 0.5
    },
    {
        "retrieved": ["A", "B"],
        "relevant": [],
        "k": 2,
        "result": 0.0
    },
])

def test_recall_at_k(parameter):
    assert recall_at_k(
        parameter["retrieved"], parameter["relevant"], parameter["k"]
    ) == parameter["result"]

@pytest.mark.parametrize("parameter", [
    {
        "retrieved": ["A", "B", "C"],
        "relevant": ["A"],
        "k": 3,
        "result": 1.0
    },
    {
        "retrieved": ["A", "B", "C"],
        "relevant": ["B"],
        "k": 3,
        "result": 0.5
    },
    {
        "retrieved": ["A", "B", "C"],
        "relevant": ["C"],
        "k": 3,
        "result": 1/3
    },
    {
        "retrieved": ["A", "B", "C"],
        "relevant": ["C", "A"],
        "k": 3,
        "result": 1.0
    },
    {
        "retrieved": ["A", "B", "C"],
        "relevant": ["C"],
        "k": 2,
        "result": 0.0
    },
])

def test_rr_at_k(parameter):
    assert reciprocal_rank_at_k(
        parameter["retrieved"], parameter["relevant"], parameter["k"]
    ) == parameter["result"]

def test_mrr_at_k():
    dataset = [
        {
            "retrieved": ["A", "B", "C"],
            "relevant": ["A"]
        },
        {
            "retrieved": ["D", "E", "F"],
            "relevant": ["E"]
        },
        {
            "retrieved": ["G", "H", "I"],
            "relevant": ["X"]
        }
    ]

    assert mrr_at_k(dataset, 3) == 0.5
    assert mrr_at_k([], 3) == 0.0
