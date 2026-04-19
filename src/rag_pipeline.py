"""Healthcare RAG pipeline — entry point and demo."""
from src.document_loader import SAMPLE_DOCUMENTS
from src.chunker import chunk_texts
from src.embedder import get_embedder
from src.vector_store import build_vector_store
from src.retriever import get_retriever
from src.generator import generate_answer


class HealthcareRAG:
    def __init__(self, k: int = 3):
        self.k = k
        self._retriever = None

    def build(self) -> None:
        print("Loading sample documents...")
        chunks = chunk_texts(SAMPLE_DOCUMENTS)
        print(f"  {len(chunks)} chunks created.")

        print("Building embeddings & vector store (first run downloads ~90 MB model)...")
        embedder = get_embedder()
        store = build_vector_store(chunks, embedder)
        self._retriever = get_retriever(store, k=self.k)
        print("  Vector store ready.\n")

    def query(self, question: str) -> str:
        if self._retriever is None:
            raise RuntimeError("Call build() before query().")
        docs = self._retriever.invoke(question)
        return generate_answer(question, docs)


def demo() -> None:
    rag = HealthcareRAG()
    rag.build()

    questions = [
        "What is the copay for generic prescription drugs?",
        "How many PTO days do I get after 3 years?",
        "Is continuous glucose monitoring covered?",
    ]

    print("=" * 60)
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {rag.query(q)}")
        print("-" * 60)


if __name__ == "__main__":
    demo()
