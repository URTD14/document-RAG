"""Configuration settings for the RAG chatbot."""

import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
SUPPORTED_FILE_TYPES = ["pdf", "txt", "csv"]


def _load_streamlit_secrets():
    try:
        import streamlit as st

        return {
            "GROQ_API_KEY": st.secrets.get("GROQ_API_KEY", ""),
            "GOOGLE_API_KEY": st.secrets.get("GOOGLE_API_KEY", ""),
        }
    except Exception:
        return {}


_streamlit_secrets = _load_streamlit_secrets()

if not GROQ_API_KEY and _streamlit_secrets.get("GROQ_API_KEY"):
    GROQ_API_KEY = _streamlit_secrets["GROQ_API_KEY"]
if not GOOGLE_API_KEY and _streamlit_secrets.get("GOOGLE_API_KEY"):
    GOOGLE_API_KEY = _streamlit_secrets["GOOGLE_API_KEY"]
