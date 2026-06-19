import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import Settings
from app.models.schemas import BuildIndexResponse, ChatRequest, ChatResponse, Citation, LLMAnswer, SystemStatus
from app.services.documents import RetrievedChunk, trim_excerpt
from app.services.indexing import build_index
from app.services.retrieval import HybridRetriever


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.retriever = HybridRetriever(settings)
        self._llm: ChatGoogleGenerativeAI | None = None
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a careful assistant for a PDF-grounded RAG system. "
                        "Answer strictly from the retrieved context.\n"
                        "Rules:\n"
                        "1. Never use outside knowledge.\n"
                        "2. If the context is insufficient, set grounded=false.\n"
                        "3. Only cite chunk ids from the allowed list.\n"
                        "4. Keep the answer concise and factual.\n"
                        "5. Return JSON only with keys: "
                        'answer, grounded, confidence, cited_chunk_ids, missing_information.'
                    ),
                ),
                (
                    "human",
                    (
                        "Question:\n{question}\n\n"
                        "Allowed chunk ids:\n{allowed_chunk_ids}\n\n"
                        "Retrieved context:\n{context}\n\n"
                        "If the answer is unsupported, say so clearly in the answer field."
                    ),
                ),
            ]
        )

        if (
            self.settings.allow_index_auto_build
            and self.settings.google_api_key
            and not self.retriever.is_ready()
        ):
            self.rebuild_index()

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            if not self.settings.google_api_key:
                raise RuntimeError("GOOGLE_API_KEY is required to query Gemini.")

            self._llm = ChatGoogleGenerativeAI(
                model=self.settings.chat_model,
                temperature=self.settings.temperature,
                google_api_key=self.settings.google_api_key,
            )
        return self._llm

    @staticmethod
    def _extract_message_text(content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "\n".join(parts)

        return str(content)

    @staticmethod
    def _cleanup_json(raw_text: str) -> str:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        return cleaned.strip()

    def _parse_llm_response(self, raw_text: str) -> LLMAnswer:
        cleaned = self._cleanup_json(raw_text)
        return LLMAnswer.model_validate(json.loads(cleaned))

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for chunk in chunks:
            page_label = chunk.page_number if chunk.page_number is not None else "Unknown"
            blocks.append(
                f"[{chunk.chunk_id}] Page {page_label} | score={chunk.score}\n{chunk.content}"
            )
        return "\n\n".join(blocks)

    def _select_citations(
        self,
        requested_chunk_ids: list[str],
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        retrieved_map = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
        citations: list[Citation] = []

        for chunk_id in requested_chunk_ids:
            chunk = retrieved_map.get(chunk_id)
            if chunk is None:
                continue
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    page=chunk.page_number,
                    score=chunk.score,
                    excerpt=trim_excerpt(chunk.content),
                )
            )

        return citations

    @staticmethod
    def _retrieval_confidence(chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        return round(sum(chunk.score for chunk in chunks) / len(chunks), 2)

    def rebuild_index(self) -> BuildIndexResponse:
        result = build_index(self.settings)
        self.retriever.refresh()
        return BuildIndexResponse(
            message="Vector index rebuilt successfully.",
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            vectorstore_dir=str(self.settings.vectorstore_dir),
            chunk_cache_path=str(self.settings.chunk_cache_path),
        )

    def system_status(self) -> SystemStatus:
        return SystemStatus(
            status="ok",
            api_name=self.settings.app_name,
            pdf_exists=self.settings.pdf_path.exists(),
            index_ready=self.retriever.is_ready(),
            chat_model=self.settings.chat_model,
            embedding_model=self.settings.embedding_model,
        )

    def _abstain(
        self,
        reason: str,
        retrieved_chunks: list[RetrievedChunk],
        notes: list[str] | None = None,
    ) -> ChatResponse:
        citations = [
            Citation(
                chunk_id=chunk.chunk_id,
                page=chunk.page_number,
                score=chunk.score,
                excerpt=trim_excerpt(chunk.content),
            )
            for chunk in retrieved_chunks[:2]
        ]
        return ChatResponse(
            answer="I could not find enough reliable support in the PDF to answer that question.",
            grounded=False,
            confidence=min(self._retrieval_confidence(retrieved_chunks), 0.35),
            answer_mode="insufficient_context",
            citations=citations,
            retrieved_chunks=len(retrieved_chunks),
            system_notes=[reason, *(notes or [])],
        )

    def answer_question(self, payload: ChatRequest) -> ChatResponse:
        if not self.retriever.is_ready():
            if self.settings.allow_index_auto_build:
                self.rebuild_index()
            else:
                raise RuntimeError("The vector index is missing. Rebuild it before querying.")

        top_k = min(payload.top_k, self.settings.max_context_chunks)
        retrieved_chunks = self.retriever.retrieve(payload.query, top_k=top_k)

        if not retrieved_chunks:
            return self._abstain("No relevant chunks were retrieved.", [])

        if retrieved_chunks[0].score < self.settings.min_retrieval_score:
            return self._abstain(
                "Top retrieval score was below the grounding threshold.",
                retrieved_chunks,
            )

        raw_response = self._get_llm().invoke(
            self._prompt.format_messages(
                question=payload.query,
                allowed_chunk_ids=", ".join(chunk.chunk_id for chunk in retrieved_chunks),
                context=self._build_context(retrieved_chunks),
            )
        )
        try:
            parsed = self._parse_llm_response(self._extract_message_text(raw_response.content))
        except Exception:
            return self._abstain(
                "The model response could not be parsed into the required JSON schema.",
                retrieved_chunks,
            )
        citations = self._select_citations(parsed.cited_chunk_ids, retrieved_chunks)

        if not parsed.grounded:
            return self._abstain(
                "The model marked the answer as ungrounded.",
                retrieved_chunks,
                parsed.missing_information,
            )

        if not citations:
            return self._abstain(
                "The model returned no valid citations from the retrieved context.",
                retrieved_chunks,
                parsed.missing_information,
            )

        grounded_chunks = [chunk for chunk in retrieved_chunks if chunk.chunk_id in {citation.chunk_id for citation in citations}]
        confidence = round(
            min(parsed.confidence, self._retrieval_confidence(grounded_chunks)),
            2,
        )

        return ChatResponse(
            answer=parsed.answer.strip(),
            grounded=True,
            confidence=confidence,
            answer_mode="grounded",
            citations=citations if payload.include_sources else [],
            retrieved_chunks=len(retrieved_chunks),
            system_notes=parsed.missing_information,
        )
