from dataclasses import dataclass
import re

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    content: str
    page_number: int | None
    source: str
    score: float
    vector_score: float
    bm25_score: float
    overlap_score: float


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def trim_excerpt(text: str, limit: int = 280) -> str:
    cleaned = normalize_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."
