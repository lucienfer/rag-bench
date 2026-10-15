def validate_dataset(dataset: dict) -> None:
    if "queries" not in dataset:
        raise ValueError(
            "dataset doesn't have \"queries\" key."
        )
    queries = dataset["queries"]
    if not isinstance(queries, list):
        raise ValueError(
            "queries is not a list"
        )

    if not queries:
        raise ValueError(
            "queries is empty"
        )

    seen_ids = set()

    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("Query must be a dict")
        if "id" not in query:
            raise ValueError(
                "query doesn't have \"id\" key."
            )
        query_id = query["id"]

        if not isinstance(query_id, str):
            raise ValueError("Query ID must be a string")

        if not query_id.strip():
            raise ValueError("id doesn't exist")

        if query_id in seen_ids:
            raise ValueError(
                "Query_id already exist"
            )

        seen_ids.add(query_id)

        if "relevant" not in query:
            raise ValueError(
                "query doesn't have \"relevant\" key."
            )

        relevant = query["relevant"]

        if not isinstance(relevant, list):
            raise ValueError("Relevant is not a list")

        result_ids = set()

        for result in relevant:
            if not isinstance(result, str):
                raise ValueError("Result must be a string")

            if not result.strip():
                raise ValueError("Result is empty")

            if result in result_ids:
                raise ValueError(
                    "result already exist"
                )

            result_ids.add(result)
