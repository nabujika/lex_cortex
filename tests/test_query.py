def test_basic_route_shape():
    payload = {
        "answer": "Grounded answer",
        "citations": [{"chunk_id": "1"}],
        "retrieved_chunks_summary": [{"chunk_id": "1"}],
        "route_used": "retriever",
    }
    assert payload["route_used"] in {"sql", "retriever", "both"}
    assert payload["citations"]

