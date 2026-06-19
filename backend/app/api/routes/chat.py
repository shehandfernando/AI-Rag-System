from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
async def query_chat(request: Request, payload: ChatRequest) -> ChatResponse:
    try:
        return await run_in_threadpool(request.app.state.rag_service.answer_question, payload)
    except Exception as exc:  # pragma: no cover - thin transport layer
        raise HTTPException(status_code=500, detail=str(exc)) from exc
