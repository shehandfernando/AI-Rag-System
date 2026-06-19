from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=2, le=8)
    include_sources: bool = True

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Query cannot be empty.")
        return cleaned


class Citation(BaseModel):
    chunk_id: str
    page: int | None = None
    score: float = Field(ge=0.0, le=1.0)
    excerpt: str = Field(min_length=1, max_length=400)


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    answer_mode: Literal["grounded", "insufficient_context"]
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: int = Field(ge=0)
    system_notes: list[str] = Field(default_factory=list)


class LLMAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=2000)
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    cited_chunk_ids: list[str] = Field(default_factory=list, max_length=4)
    missing_information: list[str] = Field(default_factory=list, max_length=5)


class BuildIndexResponse(BaseModel):
    message: str
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    vectorstore_dir: str
    chunk_cache_path: str


class SystemStatus(BaseModel):
    status: Literal["ok"]
    api_name: str
    pdf_exists: bool
    index_ready: bool
    chat_model: str
    embedding_model: str
