# Healthcare RAG

A local-first Retrieval-Augmented Generation (RAG) pipeline for querying healthcare benefits documents — built with LangChain, HuggingFace embeddings, and Chroma vector store.

## Features

- Runs fully offline using `sentence-transformers/all-MiniLM-L6-v2`
- Optional GPT-4o-mini answer generation via OpenAI
- Sample healthcare data: benefits, PTO policy, diabetes management program
- Supports PDF and DOCX document ingestion

## Project Structure

```
healthcare-rag/
├── src/
│   ├── document_loader.py  # Load PDFs, DOCX, or sample data
│   ├── chunker.py          # RecursiveCharacterTextSplitter
│   ├── embedder.py         # HuggingFace or OpenAI embeddings
│   ├── vector_store.py     # Chroma DB build & load
│   ├── retriever.py        # Similarity search retriever
│   ├── generator.py        # Answer generation (local or GPT-4o-mini)
│   └── rag_pipeline.py     # Main pipeline class + demo
└── data/
    └── chroma_db/          # Persisted vector store (git-ignored)
```

## Setup

```bash
conda create -n healthcare-rag python=3.11 -y
conda activate healthcare-rag

pip install langchain langchain-openai langchain-community langchain-huggingface \
            chromadb openai sentence-transformers python-dotenv structlog pytest
```

## Run

```bash
cd healthcare-rag
python -m src.rag_pipeline
```

The first run downloads the embedding model (~90 MB). Subsequent runs use the cached model and persisted Chroma DB.

## Use GPT-4o-mini for Answers

By default the pipeline runs in local mode and returns raw retrieved context. To generate natural language answers with GPT-4o-mini:

```bash
export USE_OPENAI=true
export OPENAI_API_KEY=sk-...
python -m src.rag_pipeline
```

## Use Your Own Documents

Place `.pdf` or `.docx` files in `data/raw/`, then call `load_documents()` instead of `SAMPLE_DOCUMENTS` in `rag_pipeline.py`:

```python
from src.document_loader import load_documents
chunks = chunk_texts(load_documents("data/raw"))
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `USE_LOCAL_EMBEDDINGS` | `true` | Use HuggingFace embeddings locally |
| `USE_OPENAI` | `false` | Use GPT-4o-mini for answer generation |
| `OPENAI_API_KEY` | — | Required when `USE_OPENAI=true` |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | Vector store persistence path |
