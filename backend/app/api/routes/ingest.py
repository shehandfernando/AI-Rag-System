import os
import shutil
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, BackgroundTasks
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import BuildIndexResponse, SystemStatus

# Keeps your existing prefix and tags
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


# --- NEW UPLOAD ENDPOINT ---
@router.post("/upload")
async def upload_pdf(
    request: Request, 
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    # 1. Validate the file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # 2. Resolve the path to the 'pdfs' folder at the root of your project
    # This traverses up from backend/app/api/routes to the main project folder
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    pdf_dir = os.path.join(base_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    # 3. Save the file to disk
    file_path = os.path.join(pdf_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # 4. Trigger your existing rag_service to rebuild the index in the background
    background_tasks.add_task(request.app.state.rag_service.rebuild_index)

    return {
        "status": "success",
        "message": f"'{file.filename}' uploaded successfully. Vectorization started in the background.",
        "file_path": file_path
    }