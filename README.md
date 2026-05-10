# 🤖 Document RAG Chatbot

> Chat with your PDFs, CSVs, text files, and URLs — powered by semantic search and LLMs.

[![Streamlit](https://img.shields.io/badge/Streamlit-1.43.0-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 📄 **PDF Chat** | Upload any PDF and ask questions about its content |
| 📊 **CSV Chat** | Upload datasets and query them conversationally |
| 📝 **Text Files** | Load `.txt` files and chat with their content |
| 🔗 **URL Chat** | Enter any webpage URL and ask about it |
| 🔍 **Semantic Search** | Gemini-powered embeddings for accurate retrieval |
| 🧠 **LLM Answers** | Groq-powered Llama for fast, contextual answers |
| 📑 **Source Citations** | Every answer shows which chunks/documents it came from |
| 🔄 **Multi-file** | Upload multiple files at once |

---

## 🏗️ Architecture

```
User Upload (PDF/TXT/CSV/URL)
        │
        ▼
   ┌─────────────┐
   │   Loader   │  ← PyPDFLoader, TextLoader, CSVLoader, URLLoader
   │  (chunking)│  ← RecursiveCharacterTextSplitter (500 chars)
   └─────┬───────┘
         ▼
   ┌─────────────┐
   │ Embeddings │  ← Gemini gemini-embedding-001 (httpx, no langchain dep)
   │  (vectors) │
   └─────┬───────┘
         ▼
   ┌─────────────┐
   │  ChromaDB  │  ← In-memory vector store
   │  (search)  │  ← Top-K similarity search
   └─────┬───────┘
         ▼
   ┌─────────────┐
   │  Groq LLM  │  ← Llama-3.3-70B-Versatile
   │  (answer)  │  ← Context + question → concise answer
   └─────────────┘
```

---

## 🚀 Live Demo

**Try it now:** 👉 [document-rag-chatbot.streamlit.app](https://document-rag-chatbot.streamlit.app)

> Upload a document → Ask questions → Get cited answers

---

## 🛠️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/URTD14/document-RAG.git
cd document-RAG
```

### 2. Get API keys

| Service | Free Tier | Sign Up |
|---------|-----------|---------|
| **Groq** | 14-day trial + free rate-limited access | [console.groq.com](https://console.groq.com) |
| **Google AI Studio** | Free tier | [aistudio.google.com](https://aistudio.google.com) |

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API keys

Create a `.env` file:

```bash
cp .env.example .env
# Edit .env with your keys:
# GROQ_API_KEY=your_groq_key
# GOOGLE_API_KEY=your_google_key
```

### 5. Run locally

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🌐 Deploy to Streamlit Cloud

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/URTD14/document-RAG.git
git push -u origin main
```

### 2. Deploy

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **New app** → Connect `document-RAG` repo
3. Expand **Advanced settings**
4. Add secrets:

```toml
GROQ_API_KEY = "gsk_your_key_here"
GOOGLE_API_KEY = "your_google_key_here"
```

5. Click **Deploy!**

---

## 📁 Project Structure

```
document-RAG/
├── app.py              # Streamlit UI
├── requirements.txt    # Dependencies
├── .env.example       # API key template
├── .gitignore
├── AGENTS.md          # Dev notes
├── src/
│   ├── __init__.py
│   ├── config.py      # Settings & secrets
│   ├── loader.py      # File loading + chunking + embeddings
│   └── chain.py       # RAG chain (Groq LLM)
└── README.md
```

---

## 🧪 How It Works

### Step 1 — Upload
Drag and drop a PDF, CSV, or TXT file into the sidebar uploader.

### Step 2 — Embed
The app chunks your document (500 chars per chunk, 50 overlap) and generates **Gemini embeddings** for each chunk. These vectors are stored in **ChromaDB** (in-memory).

### Step 3 — Ask
Type your question. The app:
1. Embeds your question with Gemini
2. Performs **semantic similarity search** in ChromaDB (top 5 chunks)
3. Sends relevant chunks + question to **Groq Llama**
4. Returns a concise, cited answer

### Step 4 — Sources
Expand the **📑 Retrieved Sources** expander to see exactly which document chunks the answer came from.

---

## ⚙️ Configuration

Edit `src/config.py` or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `LLM_TEMPERATURE` | `0.1` | Creativity (0 = deterministic) |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |

---

## 🐛 Troubleshooting

| Error | Fix |
|-------|-----|
| `Missing API keys` | Add secrets in Streamlit Cloud or `.env` file |
| `models/embedding-001 not found` | Use `gemini-embedding-001` — the custom httpx embedder handles it |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `langchain version conflict` | This app uses raw httpx for embeddings — no langchain-google-genai dependency issues |

---

## 📝 License

MIT License — feel free to use, modify, and deploy.
