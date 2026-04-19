import os

USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if USE_LOCAL:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
