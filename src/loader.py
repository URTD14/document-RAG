"""Document loaders for PDF, TXT, CSV, and URLs."""

import os
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredURLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, GOOGLE_API_KEY, CHROMA_COLLECTION_NAME


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks: list[Document]):
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set.")
    if not chunks:
        raise ValueError("No chunks provided to create vectorstore")
    embedder = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=GOOGLE_API_KEY,
    )
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=CHROMA_COLLECTION_NAME,
    )


def load_file(file_path: str, file_type: str) -> list[Document]:
    file_type = file_type.lower()
    if file_type == "pdf":
        loader = PyPDFLoader(file_path=file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path=file_path, encoding="utf-8")
    elif file_type == "csv":
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    try:
        return loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_type} file: {e}")


def load_and_process_uploaded_file(uploaded_file) -> list[Document]:
    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        docs = load_file(tmp_path, file_ext)
        return docs
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def load_url(url: str) -> list[Document]:
    try:
        loader = UnstructuredURLLoader(urls=[url])
        return loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load URL: {e}")
