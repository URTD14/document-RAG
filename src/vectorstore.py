"""ChromaDB in-memory vector store wrapper."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.embeddings import get_embedder
from src.config import CHROMA_COLLECTION_NAME


def create_vectorstore(chunks: list[Document]) -> Chroma:
    if not chunks:
        raise ValueError("No chunks provided to create vectorstore")
    embedder = get_embedder()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_memory=True,
    )


def get_retriever(top_k: int = 5):
    vs = create_vectorstore([])
    return vs.as_retriever(search_kwargs={"k": top_k})
