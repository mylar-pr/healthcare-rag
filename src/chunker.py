from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def chunk_texts(sample_docs: List[dict]) -> List[Document]:
    """Accepts list of {"content": str, "source": str} dicts."""
    docs = [
        Document(page_content=d["content"], metadata={"source": d["source"]})
        for d in sample_docs
    ]
    return chunk_documents(docs)
