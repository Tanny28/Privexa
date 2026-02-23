# ──────────────────────────────────────────────
# Query Routes — Ask questions to documents
# ──────────────────────────────────────────────
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from app.api.dependencies import require_permission
from app.rag.retriever import query_documents
from app.llm.ollama_client import get_ollama_client
from app.models.user import User
from app.config import TOP_K_RESULTS

router = APIRouter(prefix="/query", tags=["Query"])


# ─── Schemas ─────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K_RESULTS


class SourceRef(BaseModel):
    filename: str
    page_number: int
    chunk_index: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceRef]


class HealthResponse(BaseModel):
    ollama_available: bool
    model: str
    available_models: list


# ─── Endpoints ───────────────────────────────

@router.post("/", response_model=QueryResponse)
async def ask_question(
    req: QueryRequest,
    current_user: User = Depends(require_permission("query_documents")),
):
    """Ask a question and get an answer based on uploaded documents."""
    if not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    try:
        result = await query_documents(
            question=req.question,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceRef(**s) for s in result["sources"]],
    )


@router.get("/health", response_model=HealthResponse)
async def llm_health():
    """Check if the LLM (Ollama) is available."""
    llm = get_ollama_client()
    available = await llm.is_available()
    models = await llm.list_models()
    return HealthResponse(
        ollama_available=available,
        model=llm.model,
        available_models=[m.get("name", "") for m in models],
    )
