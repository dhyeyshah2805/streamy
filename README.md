# Netflix AI Agent

Netflix AI Agent helps users discover movies and TV shows through natural language and structured filters. The stack includes a **React + TypeScript** (Vite) frontend, **FastAPI** backend, **LangGraph** agent with **session memory** (`MemorySaver`), and **RAG** (sentence-transformers + FAISS) over `data/netflix_titles.csv`. Structured discovery uses **`GET /api/titles`** (no LLM); conversational recommendations use **`POST /api/chat`** (LLM + retrieval tools).

## Features

- Conversational recommendations with context carried per session (checkpointer thread id).
- Semantic search over catalog metadata via embeddings and a FAISS index.
- API-separated filtering layer: pandas-backed catalog queries independent of the LLM path.

## Quick start

Terminal 1 — API (from repo root):

```bash
uv sync
uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — UI:

```bash
cd frontend && npm install && npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). Set at least one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GROQ_API_KEY` in `.env` or your shell.

Optional: **`LLM_PROVIDER`**: `auto` (default), `openai`, `anthropic`, or `groq`.

CLI chat (same stack as `/api/chat`):

```bash
uv run python chatbot/agent.py
```

## System workflow (high level)

```
User message → LangGraph agent → (optional) RAG tool over FAISS → LLM reply
Browse UI   → GET /api/titles   → pandas filters on CSV (no LLM)
```

## Tech stack

- Python 3.13+, FastAPI, LangGraph, LangChain, sentence-transformers, FAISS  
- React, TypeScript, Vite  
- pandas / numpy for catalog handling and EDA notebooks  

## Data source

Netflix Movies and TV Shows dataset (e.g. [Kaggle — Netflix TV Shows and Movies](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies)).

## Author

**Dhyey Shah**  
📧 dhyeys2805@gmail.com  
🔗 [linkedin.com/in/dhyeys2805](https://linkedin.com/in/dhyeys2805)
