import pytest
from rag_bench.validation import validate_dataset
from rag_bench.validation import validate_retriever


def test_valid_dataset():
    valid_dataset = {
        "queries": [
            {"id": "q1", "relevant": ["A", "B"]},
            {"id": "q2", "relevant": ["C"]}
        ]
    }

    validate_dataset(valid_dataset)

def test_double_relevant():
    invalid_dataset = {
        "queries": [
            {
                "id": "q1",
                "relevant": ["A", "A"]
            }
        ]
    }

    with pytest.raises(ValueError):
            validate_dataset(invalid_dataset)

def test_invalid_id():
    invalid_dataset = {
        "queries": [
            {
                "id": "   ",
                "relevant": ["A"]
            }
        ]
    }

    with pytest.raises(ValueError):
            validate_dataset(invalid_dataset)

def test_duplicate_query_ids():
    invalid_dataset = {
        "queries": [
            {"id": "q1", "relevant": ["A"]},
            {"id": "q1", "relevant": ["B"]}
        ]
    }

    with pytest.raises(ValueError):
            validate_dataset(invalid_dataset)


def test_missing_relevant():
    invalid_dataset = {
        "queries": [
            {"id": "q1"}
        ]
    }

    with pytest.raises(ValueError):
            validate_dataset(invalid_dataset)


def test_invalid_relevant_type():
    invalid_dataset = {
        "queries": [
            {
                "id": "q1",
                "relevant": "chunk_1"
            }
        ]
    }

    with pytest.raises(ValueError):
            validate_dataset(invalid_dataset)

@pytest.mark.parametrize("dataset", [
    {"queries": []},
    {"queries": [{"id": 123, "relevant": ["A"]}]},
    {"queries": [None]},
    {"queries": [{"id": "q1", "relevant": [None]}]},
    ["q1", "q2"]
])
def test_invalid_datasets(dataset):
    with pytest.raises(ValueError):
        validate_dataset(dataset)

@pytest.mark.parametrize("retriever, should_fail", [
    ({"results": {"q1": []}}, False),
    ({"results": {}}, False),
    ({"results": {"q1": ["A", "A"]}}, True),
    ({"results": {"q1": ["A", "   "]}}, True),
    ({"results": {"q1": "A"}}, True)
])
def test_invalid_retriever(retriever, should_fail):
    if should_fail:
        with pytest.raises(ValueError):
            validate_retriever(retriever)
    else:
        validate_retriever(retriever)
