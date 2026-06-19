import json
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.services.documents import normalize_text


@dataclass(frozen=True)
class BuildIndexResult:
    page_count: int
    chunk_count: int


def _get_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required to build or query the RAG index.")

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


def load_pdf_documents(settings: Settings) -> list[Document]:
    if not settings.pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {settings.pdf_path}")

    loader = PyPDFLoader(str(settings.pdf_path))
    documents = loader.load()

    cleaned_documents: list[Document] = []
    for document in documents:
        cleaned = normalize_text(document.page_content)
        if not cleaned:
            continue

        page = document.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else None
        cleaned_documents.append(
            Document(
                page_content=cleaned,
                metadata={
                    **document.metadata,
                    "source": settings.pdf_path.name,
                    "page_number": page_number,
                },
            )
        )

    return cleaned_documents


def split_documents(settings: Settings, documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    enriched_chunks: list[Document] = []

    for index, chunk in enumerate(chunks, start=1):
        enriched_chunks.append(
            Document(
                page_content=normalize_text(chunk.page_content),
                metadata={
                    **chunk.metadata,
                    "chunk_id": f"chunk-{index:04d}",
                },
            )
        )

    return enriched_chunks


def _write_chunk_cache(settings: Settings, chunks: list[Document]) -> None:
    records = [
        {
            "chunk_id": chunk.metadata.get("chunk_id"),
            "page_number": chunk.metadata.get("page_number"),
            "source": chunk.metadata.get("source", settings.pdf_path.name),
            "content": chunk.page_content,
        }
        for chunk in chunks
    ]

    settings.chunk_cache_path.parent.mkdir(parents=True, exist_ok=True)
    settings.chunk_cache_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_index(settings: Settings) -> BuildIndexResult:
    documents = load_pdf_documents(settings)
    chunks = split_documents(settings, documents)

    vectorstore = FAISS.from_documents(chunks, _get_embeddings(settings))
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(settings.vectorstore_dir))
    _write_chunk_cache(settings, chunks)

    return BuildIndexResult(page_count=len(documents), chunk_count=len(chunks))
