import json
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from app.core.config import Settings
from app.services.documents import RetrievedChunk, tokenize
from app.services.indexing import _get_embeddings


@dataclass
class _Candidate:
    chunk_id: str
    content: str
    page_number: int | None
    source: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    overlap_score: float = 0.0


class HybridRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._vectorstore: FAISS | None = None
        self._chunk_records: list[dict] = []
        self._tokenized_chunks: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def is_ready(self) -> bool:
        return self.settings.vectorstore_dir.exists() and self.settings.chunk_cache_path.exists()

    def refresh(self) -> None:
        if not self.is_ready():
            raise FileNotFoundError("The vector index is not ready yet. Build the index first.")

        self._vectorstore = FAISS.load_local(
            str(self.settings.vectorstore_dir),
            _get_embeddings(self.settings),
            allow_dangerous_deserialization=True,
        )
        self._chunk_records = json.loads(self.settings.chunk_cache_path.read_text(encoding="utf-8"))
        self._tokenized_chunks = [tokenize(record["content"]) for record in self._chunk_records]
        self._bm25 = BM25Okapi(self._tokenized_chunks)

    def _ensure_ready(self) -> None:
        if self._vectorstore is None or self._bm25 is None:
            self.refresh()

    @staticmethod
    def _keyword_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0

        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        return len(query_set & doc_set) / max(len(query_set), 1)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self._ensure_ready()
        assert self._vectorstore is not None
        assert self._bm25 is not None

        query_tokens = tokenize(query)
        candidates: dict[str, _Candidate] = {}

        vector_results = self._vectorstore.similarity_search_with_score(
            query,
            k=self.settings.retriever_fetch_k,
        )

        for document, distance in vector_results:
            chunk_id = document.metadata.get("chunk_id")
            if not chunk_id:
                continue

            existing = candidates.get(chunk_id)
            if existing is None:
                existing = _Candidate(
                    chunk_id=chunk_id,
                    content=document.page_content,
                    page_number=document.metadata.get("page_number"),
                    source=document.metadata.get("source", self.settings.pdf_path.name),
                )
                candidates[chunk_id] = existing

            existing.vector_score = max(existing.vector_score, 1.0 / (1.0 + max(float(distance), 0.0)))
            existing.overlap_score = max(
                existing.overlap_score,
                self._keyword_overlap(query_tokens, tokenize(document.page_content)),
            )

        bm25_scores = self._bm25.get_scores(query_tokens)
        max_bm25 = max(bm25_scores) if len(bm25_scores) else 0.0
        ranked_indices = sorted(
            range(len(self._chunk_records)),
            key=lambda index: bm25_scores[index],
            reverse=True,
        )[: self.settings.retriever_fetch_k]

        for index in ranked_indices:
            record = self._chunk_records[index]
            chunk_id = record["chunk_id"]
            existing = candidates.get(chunk_id)
            if existing is None:
                existing = _Candidate(
                    chunk_id=chunk_id,
                    content=record["content"],
                    page_number=record.get("page_number"),
                    source=record.get("source", self.settings.pdf_path.name),
                )
                candidates[chunk_id] = existing

            existing.bm25_score = max(
                existing.bm25_score,
                float(bm25_scores[index]) / max_bm25 if max_bm25 > 0 else 0.0,
            )
            existing.overlap_score = max(
                existing.overlap_score,
                self._keyword_overlap(query_tokens, self._tokenized_chunks[index]),
            )

        retrieved: list[RetrievedChunk] = []
        for candidate in candidates.values():
            combined_score = (
                self.settings.vector_weight * candidate.vector_score
                + self.settings.bm25_weight * candidate.bm25_score
                + self.settings.overlap_weight * candidate.overlap_score
            )
            retrieved.append(
                RetrievedChunk(
                    chunk_id=candidate.chunk_id,
                    content=candidate.content,
                    page_number=candidate.page_number,
                    source=candidate.source,
                    score=round(min(combined_score, 1.0), 4),
                    vector_score=round(min(candidate.vector_score, 1.0), 4),
                    bm25_score=round(min(candidate.bm25_score, 1.0), 4),
                    overlap_score=round(min(candidate.overlap_score, 1.0), 4),
                )
            )

        retrieved.sort(key=lambda chunk: chunk.score, reverse=True)
        return retrieved[:top_k]
