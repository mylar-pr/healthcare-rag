import os
import pytest
from langchain_core.documents import Document

os.environ["USE_OPENAI"] = "false"


def test_local_fallback_contains_question():
    from src.generator import generate_answer
    docs = [Document(page_content="Generic drugs: $10 copay.", metadata={"source": "test"})]
    answer = generate_answer("What is the copay?", docs)
    assert "What is the copay?" in answer


def test_local_fallback_contains_context():
    from src.generator import generate_answer
    docs = [Document(page_content="Generic drugs: $10 copay.", metadata={"source": "test"})]
    answer = generate_answer("What is the copay?", docs)
    assert "$10" in answer


def test_empty_docs():
    from src.generator import generate_answer
    answer = generate_answer("What is the copay?", [])
    assert isinstance(answer, str)
