# rag-bench

**A lightweight, open-source toolkit for evaluating and comparing RAG retrieval systems.**

`rag-bench` provides a simple and reproducible way to evaluate retrieval performance using standard information retrieval metrics.

It helps developers compare retrieval configurations, identify poorly performing queries, and track experiments without requiring a database, LLM, or vector store.

## Features

- **Retrieval Evaluation** — Evaluate ranked retrieval results against a labeled dataset.
- **Standard Metrics** — Recall@K, Precision@K, and Mean Reciprocal Rank (MRR@K).
- **Query-Level Analysis** — Inspect individual query performance.
- **Retriever Comparison** — Compare retrieval results from different configurations.
- **Dataset Validation** — Detect invalid datasets and duplicate query identifiers.
- **Experiment Tracking** — Save experiment configurations and evaluation results as JSON.
- **Command-Line Interface** — Run evaluations directly from your terminal.
- **Lightweight** — No external database, LLM, or vector store required.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Installation

Clone the repository:

```bash
git clone https://github.com/lucienfer/rag-bench
cd rag-bench
```

Install the project and its dependencies:

```bash
uv sync
```

Verify the installation:

```bash
uv run rag-bench --help
```

## Quick Start

`rag-bench` evaluates retrieval results against a reference dataset.

### 1. Create an Evaluation Dataset

Create `datasets/evaluation.json`:

```json
{
  "queries": [
    {
      "id": "q1",
      "relevant": ["chunk_1", "chunk_3"]
    },
    {
      "id": "q2",
      "relevant": ["chunk_5"]
    }
  ]
}
```

Each query contains a unique identifier and a list of relevant chunk IDs.

### 2. Create Retrieval Results

Create `datasets/retriever.json`:

```json
{
  "name": "example-retriever",
  "results": {
    "q1": ["chunk_1", "chunk_2", "chunk_3"],
    "q2": ["chunk_4", "chunk_5", "chunk_6"]
  }
}
```

Results must be ordered from most relevant to least relevant.

Every query in the evaluation dataset must have a corresponding entry in the retrieval results.

An empty result list is valid:

```json
{
  "results": {
    "q1": []
  }
}
```

It indicates that the retriever returned no documents.

### 3. Run an Evaluation

```bash
uv run rag-bench evaluate \
  --dataset datasets/evaluation.json \
  --results datasets/retriever.json \
  --k 3
```

Example output:

```text
RAG Benchmark
-----------------------------
Queries evaluated : 2

Recall@3      : 1.000
Precision@3   : 0.500
MRR@3         : 0.750
```

## Evaluation Metrics

`rag-bench` currently supports three retrieval metrics.

### Recall@K

Measures the proportion of relevant documents retrieved within the top K results.

**Formula:**

Recall@K = Relevant documents retrieved in top K / Total relevant documents

A higher Recall@K indicates that fewer relevant documents are missed.

### Precision@K

Measures the proportion of relevant documents among the top K results.

**Formula:**

Precision@K = Relevant documents retrieved in top K / K

Precision@K is useful for evaluating how much irrelevant information a retriever introduces into the context.

When fewer than K results are returned, K is still used as the denominator.

### MRR@K

Mean Reciprocal Rank measures how highly the first relevant document is ranked.

For each query:

**Formula:**

RR@K = 1 / Rank of the first relevant document

If no relevant document appears within the top K results, the reciprocal rank is zero.

MRR@K is the average reciprocal rank across all evaluated queries.

Unlike Recall@K, MRR@K only considers the position of the first relevant result.

## Comparing Retrievers

To compare two retrieval configurations, prepare separate result files using the same evaluation dataset.

For example:

```text
datasets/
├── evaluation.json
├── retriever_a.json
└── retriever_b.json
```

Run evaluations independently:

```bash
uv run rag-bench evaluate \
  --dataset datasets/evaluation.json \
  --results datasets/retriever_a.json \
  --k 5
```

```bash
uv run rag-bench evaluate \
  --dataset datasets/evaluation.json \
  --results datasets/retriever_b.json \
  --k 5
```

Compare their Recall@K, Precision@K, and MRR@K scores.

For meaningful comparisons, use the same reference dataset, corpus version, and evaluation settings.

The Python evaluation engine also supports query-level results and programmatic comparison.

## Experiment Tracking

`rag-bench` includes a lightweight JSON-based experiment tracking utility.

Experiments can store:

- Experiment ID and timestamp.
- Embedding model configuration.
- Chunk size and overlap.
- Corpus and dataset versions.
- Evaluation metrics.
- Query-level evaluation results.

Each experiment is saved as a separate JSON file.

This makes it possible to preserve evaluation results and compare retrieval configurations over time without requiring an external tracking service.

Experiment persistence is currently available through Python; the CLI does not yet expose an experiment-saving command.

## Dataset Validation

Evaluation datasets are validated before processing.

The validator checks that:

- The dataset contains a non-empty `queries` list.
- Every query has a unique, non-empty string identifier.
- Every query contains a `relevant` list.
- Relevant document identifiers are non-empty strings.
- Relevant identifiers are not duplicated within a query.

Missing retrieval results are treated as errors rather than silently ignored.

An empty list of retrieved documents is valid and produces zero retrieval scores.

## Running Tests

Install development dependencies:

```bash
uv sync --dev
```

Run the test suite:

```bash
uv run pytest
```

For detailed output:

```bash
uv run pytest -v
```

The test suite covers retrieval metrics, dataset validation, evaluation aggregation, and retriever comparison.

## Project Structure

```text
rag-bench/
├── pyproject.toml
├── README.md
├── src/
│   └── rag_bench/
│       ├── __init__.py
│       ├── cli.py
│       ├── evaluate_retriever.py
│       ├── validation.py
│       └── metrics/
│           ├── __init__.py
│           ├── recall.py
│           ├── precision.py
│           └── ranking.py
├── datasets/
│   ├── evaluation.json
│   ├── retriever_a.json
│   └── retriever_b.json
└── tests/
    ├── test_metrics.py
    ├── test_validation.py
    └── test_evaluator.py
```

## Limitations

`rag-bench` is an early-stage retrieval evaluation toolkit.

### Chunk-Level Annotations

Relevance judgments currently reference chunk identifiers.

Comparing different chunking strategies may require remapping relevance annotations, since chunk identifiers can change between index generations.

### Annotation Quality

Evaluation results depend on the completeness and correctness of relevance judgments.

Unlabeled but relevant passages may be incorrectly considered irrelevant.

### Retrieval Only

The toolkit evaluates document retrieval, not the quality of answers generated by an LLM.

Answer correctness, faithfulness, and citation quality are outside the current scope.

### Small Evaluation Datasets

Metrics calculated on small datasets may not accurately represent real-world retrieval performance.

Larger and more representative benchmarks are recommended before drawing conclusions about model quality.

### Queries Without Relevant Documents

The current metrics use a zero-score convention for empty relevance sets.

Such queries should be evaluated separately when measuring retrieval abstention or no-answer behavior.

## Roadmap

Potential future improvements include:

- Automated comparison reports.
- Configurable evaluation metrics.
- nDCG@K and additional ranking metrics.
- Experiment tracking through the CLI.
- Synthetic evaluation dataset generation.
- Article-level and passage-level relevance annotations.
- Integration with RAG pipelines and retrieval APIs.
- LLM answer evaluation.

## License

License to be determined.
