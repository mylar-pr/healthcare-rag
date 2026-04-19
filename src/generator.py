import os
from typing import List
from langchain_core.documents import Document

USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"

SYSTEM_PROMPT = (
    "You are a helpful healthcare benefits assistant. "
    "Answer questions using only the provided context. "
    "If the context does not contain the answer, say so clearly."
)


def generate_answer(question: str, context_docs: List[Document]) -> str:
    context = "\n\n".join(doc.page_content for doc in context_docs)
    if USE_OPENAI:
        return _openai_answer(question, context)
    return _local_answer(question, context)


def _openai_answer(question: str, context: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def _local_answer(question: str, context: str) -> str:
    return (
        f"[Local mode — set USE_OPENAI=true for GPT-4o-mini]\n\n"
        f"Question: {question}\n\n"
        f"Relevant context retrieved:\n{context}"
    )
