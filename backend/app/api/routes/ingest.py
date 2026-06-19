from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import BuildIndexResponse, SystemStatus

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.get("/status", response_model=SystemStatus)
def ingest_status(request: Request) -> SystemStatus:
    return request.app.state.rag_service.system_status()


@router.post("/rebuild", response_model=BuildIndexResponse)
async def rebuild_index(request: Request) -> BuildIndexResponse:
    try:
        return await run_in_threadpool(request.app.state.rag_service.rebuild_index)
    except Exception as exc:  # pragma: no cover - thin transport layer
        raise HTTPException(status_code=500, detail=str(exc)) from exc
