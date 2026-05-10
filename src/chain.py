"""Groq RAG chain — semantic search + LLM answer."""

from typing import Literal
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from src.config import (
    GROQ_API_KEY,
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K_RESULTS,
)


def query_documents(vectorstore: Chroma, question: str) -> tuple[str, list[Document]]:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set.")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set.")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    docs = retriever.invoke(question)

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[Source {i + 1}]: {doc.page_content}" for i, doc in enumerate(docs)
        )

    system_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question.
If the answer is not in the context, say you don't know. Be concise and cite relevant parts."""

    context = format_docs(docs)
    full_prompt = (
        f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    answer = llm.invoke(full_prompt).content
    return answer, docs
