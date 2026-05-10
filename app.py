"""Streamlit UI for the RAG Chatbot."""

import streamlit as st
from src import loader
from src import chain as rag_chain
from src.config import GROQ_API_KEY, GOOGLE_API_KEY, SUPPORTED_FILE_TYPES

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot — Chat with Your Documents")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False


def validate_keys():
    errors = []
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY")
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY")
    if errors:
        st.error(
            f"Missing API keys: {', '.join(errors)}. Add them via Streamlit secrets or .env file."
        )
        st.stop()


@st.cache_resource
def build_vectorstore(_docs):
    chunks = loader.chunk_documents(_docs)
    return loader.create_vectorstore(chunks)


def add_sources_to_history(question: str, answer: str, docs: list):
    sources = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            page = doc.metadata.get("page", None)
            label = f"{src} (page {page})" if page is not None else src
            sources.append(label)
    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
    )


with st.sidebar:
    st.header("📂 Upload Documents")
    validate_keys()

    tab = st.radio(
        "Input type", ["📄 Upload Files", "🔗 URL"], label_visibility="collapsed"
    )

    if tab == "📄 Upload Files":
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or CSV",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            help="Upload your documents to chat with them",
        )
        if uploaded_files:
            with st.spinner("Processing files..."):
                all_docs = []
                for f in uploaded_files:
                    try:
                        docs = loader.load_and_process_uploaded_file(f)
                        all_docs.extend(docs)
                    except Exception as e:
                        st.error(f"Error loading {f.name}: {e}")
                if all_docs:
                    try:
                        vs = build_vectorstore(all_docs)
                        st.session_state.vectorstore = vs
                        st.session_state.docs_loaded = True
                        total_chunks = len(loader.chunk_documents(all_docs))
                        st.success(
                            f"Loaded {len(uploaded_files)} file(s) — {total_chunks} chunks embedded!"
                        )
                    except Exception as e:
                        st.error(f"Embedding failed: {e}")
        else:
            st.info("Upload files to get started.")

    elif tab == "🔗 URL":
        url_input = st.text_input(
            "Enter URL", placeholder="https://example.com/article"
        )
        if st.button("Load URL") and url_input:
            with st.spinner("Fetching URL content..."):
                try:
                    docs = loader.load_url(url_input)
                    if docs:
                        vs = build_vectorstore(docs)
                        st.session_state.vectorstore = vs
                        st.session_state.docs_loaded = True
                        total_chunks = len(loader.chunk_documents(docs))
                        st.success(
                            f"Loaded {len(docs)} sections — {total_chunks} chunks embedded!"
                        )
                except Exception as e:
                    st.error(f"Failed to load URL: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.docs_loaded = False
        st.rerun()

    st.divider()
    st.caption(
        "**How it works**\n1. Upload files or enter URL\n2. Ask questions below\n3. AI answers using your docs"
    )


for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {msg['question']}")
    with st.chat_message("assistant"):
        st.markdown(msg["answer"])
        if msg["sources"]:
            with st.expander("📑 Sources"):
                for s in msg["sources"]:
                    st.caption(s)


if prompt := st.chat_input("Ask something about your documents..."):
    if st.session_state.vectorstore is None:
        st.warning("Please upload documents or enter a URL first.")
    else:
        with st.chat_message("user"):
            st.markdown(f"**You:** {prompt}")
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, docs = rag_chain.query_documents(
                        st.session_state.vectorstore, prompt
                    )
                    st.markdown(answer)
                    add_sources_to_history(prompt, answer, docs)
                    with st.expander("📑 Retrieved Sources"):
                        for i, doc in enumerate(docs):
                            src = doc.metadata.get("source", "unknown")
                            page = doc.metadata.get("page", None)
                            label = f"Chunk {i + 1}" + (
                                f" — {src} p.{page}"
                                if page is not None
                                else f" — {src}"
                            )
                            st.caption(f"**{label}:** {doc.page_content[:300]}...")
                except Exception as e:
                    st.error(f"Error: {e}")
