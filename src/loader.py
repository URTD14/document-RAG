"""Document loaders for PDF, TXT, CSV, and URLs."""

import os
import tempfile
import httpx
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredURLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, GOOGLE_API_KEY, CHROMA_COLLECTION_NAME


class GeminiEmbeddings(Embeddings):
    def __init__(
        self, api_key: str, model: str = "text-embedding-004", batch_size: int = 100
    ):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            payload = {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": t} for t in batch]},
                "task_type": "RETRIEVAL_DOCUMENT",
            }
            url = f"{self.base_url}/{self.model}:embedContent?key={self.api_key}"
            with httpx.Client(timeout=60) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if "embeddings" in data:
                    embeddings.extend([list(e["values"]) for e in data["embeddings"]])
                elif "embedding" in data:
                    embeddings.extend([list(data["embedding"]["values"])])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks: list[Document]):
    from langchain_chroma import Chroma

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set.")
    if not chunks:
        raise ValueError("No chunks provided to create vectorstore")
    embedder = GeminiEmbeddings(api_key=GOOGLE_API_KEY, model="gemini-embedding-001")
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
