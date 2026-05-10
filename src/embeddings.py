"""Gemini embeddings wrapper for document chunk vectors."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import GOOGLE_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION


def get_embedder() -> GoogleGenerativeAIEmbeddings:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set. Add it to .env or Streamlit secrets.")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        model_version="001",
        task_type="retrieval_document",
    )
