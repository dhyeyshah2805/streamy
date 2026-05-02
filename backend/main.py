import asyncio
from contextlib import asynccontextmanager

import backend.config  # noqa: F401 — load_dotenv before services read env

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import chat, titles
from backend.services.rag_service import get_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm embedding index + FAISS so the first chat request is responsive.
    try:

        def _warm() -> None:
            get_rag().ensure_loaded()

        await asyncio.to_thread(_warm)
    except Exception:
        # CSV missing or model download issue — surface on first real request
        pass
    yield


app = FastAPI(
    title="Netflix AI Agent",
    description="FastAPI layer: `/api/titles` (filters) vs `/api/chat` (LLM + RAG).",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(titles.router, prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
