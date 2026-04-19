from src.chunker import chunk_texts, chunk_documents
from langchain_core.documents import Document


SAMPLE = [
    {"content": "Generic drugs cost $10 copay. Brand-name drugs cost $35 copay.", "source": "benefits.txt"},
    {"content": "PTO accrual: 0-2 years earns 15 days per year.", "source": "pto.txt"},
]


def test_chunk_texts_returns_documents():
    chunks = chunk_texts(SAMPLE)
    assert len(chunks) > 0
    assert all(hasattr(c, "page_content") for c in chunks)


def test_chunk_texts_preserves_source():
    chunks = chunk_texts(SAMPLE)
    sources = {c.metadata["source"] for c in chunks}
    assert "benefits.txt" in sources
    assert "pto.txt" in sources


def test_chunk_documents_respects_size():
    doc = Document(page_content="word " * 300, metadata={"source": "test"})
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=0)
    assert all(len(c.page_content) <= 120 for c in chunks)


def test_chunk_ids_assigned():
    chunks = chunk_texts(SAMPLE)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))
