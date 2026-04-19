import os
import pytest

os.environ["USE_LOCAL_EMBEDDINGS"] = "true"
os.environ["USE_OPENAI"] = "false"
os.environ["CHROMA_PERSIST_DIR"] = "data/test_chroma_db"


@pytest.fixture(scope="module")
def rag():
    from src.rag_pipeline import HealthcareRAG
    pipeline = HealthcareRAG(k=2)
    pipeline.build()
    return pipeline


def test_build_succeeds(rag):
    assert rag._retriever is not None


def test_query_returns_string(rag):
    answer = rag.query("What is the copay for generic drugs?")
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_query_retrieves_relevant_context(rag):
    answer = rag.query("What is the copay for generic drugs?")
    assert "$10" in answer


def test_query_pto(rag):
    answer = rag.query("How many PTO days after 3 years?")
    assert "20" in answer


def test_query_diabetes(rag):
    answer = rag.query("Is continuous glucose monitoring covered?")
    assert "90%" in answer or "CGM" in answer


def test_query_before_build_raises():
    from src.rag_pipeline import HealthcareRAG
    pipeline = HealthcareRAG()
    with pytest.raises(RuntimeError):
        pipeline.query("test")
