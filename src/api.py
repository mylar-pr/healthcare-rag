"""FastAPI backend for the Healthcare RAG pipeline."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag_pipeline import HealthcareRAG

rag: HealthcareRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = HealthcareRAG()
    rag.build()
    yield


app = FastAPI(title="Healthcare RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str


@app.get("/health")
def health():
    return {"status": "ok", "ready": rag is not None}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    answer = rag.query(request.question)
    return QueryResponse(question=request.question, answer=answer)
