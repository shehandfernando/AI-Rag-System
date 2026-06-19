from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/query", response_model=ChatResponse)
async def query_chat(request: Request, payload: ChatRequest) -> ChatResponse:
    try:
        return await run_in_threadpool(request.app.state.rag_service.answer_question, payload)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# --- NEW STREAM ENDPOINT ---
@router.post("/stream")
async def stream_chat(request: Request, payload: ChatRequest):
    try:
        # We grab the generator function from our RAG service
        generator = request.app.state.rag_service.stream_answer_question(payload)
        
        # We return a StreamingResponse, telling the browser to expect Server-Sent Events (SSE)
        return StreamingResponse(
            generator, 
            media_type="text/event-stream"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc