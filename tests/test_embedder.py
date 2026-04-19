import os
import pytest

os.environ["USE_LOCAL_EMBEDDINGS"] = "true"


def test_embedder_returns_embeddings():
    from src.embedder import get_embedder
    embedder = get_embedder()
    result = embedder.embed_query("What is the copay for generic drugs?")
    assert isinstance(result, list)
    assert len(result) == 384  # all-MiniLM-L6-v2 output dimension


def test_embedder_documents():
    from src.embedder import get_embedder
    embedder = get_embedder()
    results = embedder.embed_documents(["copay", "PTO policy"])
    assert len(results) == 2
    assert all(len(r) == 384 for r in results)
