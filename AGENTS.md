# RAG Chatbot Development Notes

## API Keys
- `GROQ_API_KEY` — Get from https://console.groq.com
- `GOOGLE_API_KEY` — Get from https://aistudio.google.com

## Local Development
```bash
cp .env.example .env
# Edit .env with your keys
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Cloud)
1. Push to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repo, deploy
4. Add secrets:
   - `GROQ_API_KEY` = your groq key
   - `GOOGLE_API_KEY` = your google ai studio key

## Architecture
- `src/loader.py` — PDF/TXT/CSV/URL → raw text
- `src/embeddings.py` — Gemini embeddings → vectors
- `src/vectorstore.py` — ChromaDB (in-memory)
- `src/chain.py` — Groq RAG chain
- `app.py` — Streamlit UI

## Data Flow
Upload → Chunk → Embed → Store → Query → Retrieve Top-K → Groq Answer
