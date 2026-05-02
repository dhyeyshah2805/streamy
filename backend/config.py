import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NETFLIX_CSV_PATH = Path(
    os.getenv("NETFLIX_CSV_PATH", str(PROJECT_ROOT / "data" / "netflix_titles.csv"))
)

RAG_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RAG_TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "12"))
RAG_RETRIEVE_MULTIPLIER = int(os.getenv("RAG_RETRIEVE_MULTIPLIER", "4"))
