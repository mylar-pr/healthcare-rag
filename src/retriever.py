from langchain_community.vectorstores import Chroma


def get_retriever(vector_store: Chroma, k: int = 3):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
