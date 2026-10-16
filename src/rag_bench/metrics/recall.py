def recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int
) -> float:
    relevant_set = set(relevant)

    if k <= 0:
        raise ValueError("k must be positive")

    if len(relevant) == 0 :
        return 0.0
    retrieved_set = set(retrieved[:k])
    return len(retrieved_set & relevant_set) / len(relevant_set)
