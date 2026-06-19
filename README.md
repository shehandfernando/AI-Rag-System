# RAG System

This repository started as a notebook-first RAG prototype in `rag_notebook.ipynb`. The notebook loads `pdfs/mathematics.pdf`, chunks it, embeds it with Gemini, stores vectors in FAISS, retrieves the top matches, and answers with a constrained prompt. That same flow was then copied into a basic Streamlit app in `app.py`.

The project is now split into a production-leaning full-stack structure:

```text
rag_chatbot/
├─ backend/
│  ├─ app/
│  │  ├─ api/routes/
│  │  ├─ core/
│  │  ├─ models/
│  │  └─ services/
│  ├─ data/
│  ├─ .env.example
│  └─ requirements.txt
├─ frontend/
│  ├─ app/
│  ├─ components/
│  ├─ lib/
│  ├─ .env.example
│  └─ package.json
├─ pdfs/
│  └─ mathematics.pdf
├─ vectorstores/
│  └─ mathematics_faiss_gemini/
├─ rag_notebook.ipynb
├─ app.py
└─ README.md
```

## What The Notebook Already Did

`rag_notebook.ipynb` currently does these steps:

1. Loads the mathematics PDF with `PyPDFLoader`.
2. Splits text with `RecursiveCharacterTextSplitter`.
3. Builds a FAISS vector store using `models/gemini-embedding-001`.
4. Retrieves the top matching chunks for a question.
5. Sends retrieved context to `gemini-2.5-flash`.
6. Returns an answer with a simple "only use the context" prompt.
7. Writes the same logic into the Streamlit prototype in `app.py`.

That notebook is a solid proof of concept, but it still mixes ingestion, retrieval, generation, and UI in one place.

## New Architecture

### Backend

The backend is a FastAPI service that owns the RAG logic:

- Validated request and response schemas with Pydantic.
- PDF ingestion and chunk caching.
- FAISS vector retrieval plus BM25 lexical retrieval.
- Hybrid score fusion to reduce missed relevant chunks.
- Strict JSON answer contract from the LLM.
- Citation validation so only retrieved chunks can be cited.
- Abstention logic when evidence is weak or the model output is malformed.

Important files:

- `backend/app/main.py`
- `backend/app/api/routes/chat.py`
- `backend/app/api/routes/ingest.py`
- `backend/app/models/schemas.py`
- `backend/app/services/indexing.py`
- `backend/app/services/retrieval.py`
- `backend/app/services/rag.py`

### Frontend

The frontend is a minimal Next.js app focused on usability:

- Clean single-page chat experience.
- Status panel for API and index health.
- Citation cards for every grounded answer.
- Rebuild-index action.
- A deliberately minimal visual language so the model output stays central.

Important files:

- `frontend/app/page.tsx`
- `frontend/components/chat-shell.tsx`
- `frontend/lib/api.ts`
- `frontend/app/globals.css`

## Hallucination Reduction Strategy

This version is designed to reduce hallucination much more aggressively than the notebook prototype:

- Retrieval is hybrid, not dense-only.
- Temperature is fixed at `0`.
- The prompt forbids outside knowledge.
- The model must return structured JSON.
- The backend validates the model output.
- The backend rejects answers without valid citations.
- Low-confidence retrieval triggers an abstain response instead of a guessed answer.
- Chunk cache persistence helps debug retrieval quality over time.

## Backend Setup

Create `backend/.env` from `backend/.env.example`:

```env
GOOGLE_API_KEY=your_google_api_key
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

Run the API from the `backend/` folder:

```bash
cd backend
uvicorn app.main:app --reload
```

Useful endpoints:

- `GET /health`
- `POST /api/v1/chat/query`
- `GET /api/v1/ingest/status`
- `POST /api/v1/ingest/rebuild`

## Frontend Setup

Create `frontend/.env.local` from `frontend/.env.example`:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Then install and run:

```bash
cd frontend
npm install
npm run dev
```

## Suggested Next Improvements

- Add a small evaluation dataset for regression testing.
- Add reranking after hybrid retrieval.
- Add streaming responses for the frontend.
- Add observability around retrieval score distributions and abstain rates.
- Add unit tests for parser fallback, citation enforcement, and rebuild flow.

## Legacy Prototype

`rag_notebook.ipynb` and `app.py` are still useful as the original prototype and reference implementation. The new `backend/` and `frontend/` folders are the recommended path forward.
