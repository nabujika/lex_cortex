from app.ingestion.chunker import chunk_case_text, infer_chunk_type


def test_chunker_creates_chunks():
    pages = [{"page_number": 1, "text": "Facts: Alpha " * 300}]
    chunks = chunk_case_text(case_id="case-1", pages=pages)
    assert chunks
    assert all(chunk.case_id == "case-1" for chunk in chunks)


def test_chunk_type_inference():
    assert infer_chunk_type("Dissent: the minority opinion") == "dissent"
    assert infer_chunk_type("Facts of the case are as follows") == "facts"

