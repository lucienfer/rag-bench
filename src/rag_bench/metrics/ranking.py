
def reciprocal_rank_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int
) -> float:
    if k <= 0 :
        raise ValueError("k must be positive")

    relevant_set = set(relevant)

    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant_set:
            return 1 / rank
    return 0.0


def mrr_at_k(
    dataset: list[dict],
    k: int
) -> float:
    result = 0

    if not dataset:
        return 0.0

    if k <= 0 :
        raise ValueError("k must be positive")

    for value in dataset :
        result += reciprocal_rank_at_k(value["retrieved"], value["relevant"], k)

    return result / len(dataset)
