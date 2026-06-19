import json
from pathlib import Path
import pytest
from dotenv import load_dotenv

load_dotenv()

from app.core.config import Settings
from app.services.rag import RAGService

# Load the test cases from your JSON file
EVAL_FILE = Path(__file__).parent / "eval_dataset.json"
TEST_CASES = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

@pytest.fixture(scope="module")
def rag_service():
    """Initializes the RAG service once for all tests."""
    settings = Settings()
    service = RAGService(settings)
    
    # Ensure the index is built before testing
    if not service.retriever.is_ready():
        service.rebuild_index()
        
    return service

@pytest.mark.parametrize("case", TEST_CASES)
def test_retrieval_accuracy(rag_service: RAGService, case: dict):
    """
    Tests if the expected text snippet is found in the top retrieved chunks.
    """
    question = case["question"]
    expected_snippet = case["expected_text_snippet"].lower()
    
    # Run the retrieval pipeline (which now includes FlashRank!)
    # We fetch the top 3 chunks to ensure high precision
    retrieved_chunks = rag_service.retriever.retrieve(query=question, top_k=3)
    
    # Combine all retrieved text into one massive string for easy searching
    combined_context = " ".join([chunk.content.lower() for chunk in retrieved_chunks])
    
    # The actual test assertion
    assert expected_snippet in combined_context, (
        f"Regression Failure: The system failed to retrieve the necessary context for the question: '{question}'"
    )