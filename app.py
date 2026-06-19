import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

PDF_PATH = Path("pdfs/mathematics.pdf")
VECTORSTORE_DIR = Path("vectorstores/mathematics_faiss_gemini")
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

st.set_page_config(page_title="Mathematics Chatbot", layout="wide")
st.title("Mathematics PDF Chatbot")
st.caption("Ask questions about pdfs/mathematics.pdf using Gemini and FAISS")

api_key = st.sidebar.text_input("Google API Key (optional override)", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

if not os.environ.get("GOOGLE_API_KEY"):
    st.info("Set GOOGLE_API_KEY in .env or enter it in the sidebar to continue.")
    st.stop()

if not PDF_PATH.exists():
    st.error(f"PDF not found: {PDF_PATH}")
    st.stop()


def page_label(doc):
    page = doc.metadata.get("page")
    return page + 1 if isinstance(page, int) else page


def load_pdf_documents(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def create_vectorstore(pdf_path: Path, vectorstore_dir: Path):
    documents = load_pdf_documents(pdf_path)
    chunks = split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_dir))
    return len(documents), len(chunks)


def load_vectorstore(vectorstore_dir: Path):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=False)
def get_rag_components():
    vectorstore = load_vectorstore(VECTORSTORE_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful mathematics tutor.
Answer the user's question using only the retrieved context below.
If the answer is not in the context, say you could not find it in the PDF.

Context:
{context}

Question:
{question}
""".strip()
    )
    return retriever, llm, prompt


def format_docs(docs):
    return "\n\n".join(
        f"Source {i + 1} | Page {page_label(doc)}\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


def ask_question(question: str, retriever, llm, prompt):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = llm.invoke(prompt.format_messages(context=context, question=question)).content
    return answer, docs


if "messages" not in st.session_state:
    st.session_state.messages = []

left, right = st.columns([1, 1])

with left:
    if st.button("Create / Refresh Vector Store"):
        with st.spinner("Building vector store..."):
            page_count, chunk_count = create_vectorstore(PDF_PATH, VECTORSTORE_DIR)
            get_rag_components.clear()
        st.success(f"Vector store ready. Pages: {page_count}, Chunks: {chunk_count}")

with right:
    st.write(f"Chat model: `{CHAT_MODEL}` | Embedding model: `{EMBEDDING_MODEL}`")

if not VECTORSTORE_DIR.exists():
    st.warning("Build the vector store first using the button above.")
    st.stop()

retriever, llm, prompt = get_rag_components()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(message["sources"])

user_question = st.chat_input("Ask a question about the mathematics PDF")
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, docs = ask_question(user_question, retriever, llm, prompt)
        pages = sorted({str(page_label(doc)) for doc in docs})
        sources = "Retrieved pages: " + ", ".join(pages)
        st.markdown(answer)
        st.caption(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
