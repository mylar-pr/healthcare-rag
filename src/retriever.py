from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List


def get_retriever(vector_store: Chroma, k: int = 3):
    base = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 4})
    return base
