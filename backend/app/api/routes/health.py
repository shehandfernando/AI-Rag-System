from fastapi import APIRouter, Request

from app.models.schemas import SystemStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=SystemStatus)
def health_check(request: Request) -> SystemStatus:
    return request.app.state.rag_service.system_status()
