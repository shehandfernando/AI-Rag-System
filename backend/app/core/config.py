from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    app_name: str = "Mathematics RAG API"
    api_prefix: str = "/api/v1"
    environment: Literal["development", "production", "test"] = "development"

    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")

    pdf_path: Path = PROJECT_ROOT / "pdfs" / "mathematics.pdf"
    vectorstore_dir: Path = PROJECT_ROOT / "vectorstores" / "mathematics_faiss_gemini"
    chunk_cache_path: Path = BACKEND_ROOT / "data" / "chunks.json"

    embedding_model: str = "models/gemini-embedding-001"
    chat_model: str = "gemini-2.5-flash"

    chunk_size: int = Field(default=1000, ge=300, le=2000)
    chunk_overlap: int = Field(default=150, ge=0, le=400)

    retriever_k: int = Field(default=4, ge=2, le=8)
    retriever_fetch_k: int = Field(default=10, ge=4, le=24)
    max_context_chunks: int = Field(default=4, ge=2, le=8)

    vector_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    overlap_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    min_retrieval_score: float = Field(default=0.18, ge=0.0, le=1.0)

    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    allow_index_auto_build: bool = True

    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
