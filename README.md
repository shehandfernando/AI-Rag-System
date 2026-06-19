# Full-Stack AI RAG System with Real-Time Streaming

A production-ready Retrieval-Augmented Generation (RAG) application designed to accurately extract, retrieve, and synthesize information from complex academic documents. 

This project allows users to dynamically upload PDFs and query them through a highly responsive chat interface. It utilizes a multi-stage retrieval pipeline to ensure the LLM strictly grounds its answers in the provided context, preventing hallucinations.

## ✨ Key Features

* **Real-Time Streaming (SSE):** Implements Server-Sent Events to stream LLM tokens directly to the Next.js UI character-by-character, providing a native ChatGPT-like experience without blocking requests.
* **Hybrid Retrieval Pipeline:** Combines **FAISS** vector similarity search (for semantic meaning) with **BM25** keyword matching (for exact terminology) to cast a wide, highly accurate net for relevant document chunks.
* **Cross-Encoder Reranking:** Utilizes the **FlashRank** model to re-score and filter the initial retrieved chunks, ensuring pinpoint precision before sending context to the LLM.
* **Smart Intent Routing:** Features an "Abstract Bypass" router that detects broad summary requests and intelligently feeds introductory cache data to the LLM, bypassing the standard vector search limits.
* **Dynamic Document Ingestion:** Users can upload new PDFs directly through the web UI. The backend automatically parses, chunks, and embeds the text in the background.
* **Automated Regression Testing:** Includes a robust `pytest` suite designed to continuously validate the accuracy of the retrieval pipeline against expected document facts.

## 🛠️ Tech Stack

**Frontend:**
* Next.js (React)
* TypeScript
* CSS (Custom modular styling)

**Backend:**
* Python (FastAPI, Uvicorn)
* LangChain & LangGraph
* Pytest (Regression testing)

**AI & Machine Learning:**
* **LLM & Embeddings:** Google Gemini (`gemini-embedding-1.0` and Chat models)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Keyword Indexing:** BM25 (Okapi)
* **Reranking:** FlashRank (Cross-Encoder)

---

## 🚀 Getting Started

### Prerequisites
* Node.js (v18+)
* Python (3.11+)
* A free [Google Gemini API Key](https://aistudio.google.com/)

### 1. Backend Setup
Navigate to the backend directory and install the dependencies:

```bash
cd backend
pip install -r requirements.txt
pip install flashrank python-multipart
```

Create a `.env` file in the `backend/` directory and add your API key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Start the FastAPI server:

```bash
python -m uvicorn app.main:app --reload
```
*The backend will be running at `http://127.0.0.1:8000`*

### 2. Frontend Setup
Open a new terminal window, navigate to the frontend directory, and install the dependencies:

```bash
cd frontend
npm install
```

Start the Next.js development server:

```bash
npm run dev
```
*The frontend will be running at `http://localhost:3000`*

## 🧪 Running Tests
To ensure the retrieval pipeline is accurately finding the expected context, run the regression suite from the `backend/` directory:

```bash
python -m pytest tests/test_retrieval.py -v
```