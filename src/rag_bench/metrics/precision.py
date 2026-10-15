def precision_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = set(retrieved[:k])
    relevant_set = set(relevant)

    return  len(top_k & relevant_set) / k
