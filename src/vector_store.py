import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")


def build_vector_store(chunks: List[Document], embedder) -> Chroma:
    return Chroma.from_documents(chunks, embedder, persist_directory=PERSIST_DIR)


def load_vector_store(embedder) -> Chroma:
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedder)
