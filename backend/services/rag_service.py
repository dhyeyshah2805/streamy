from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from backend.config import (
    NETFLIX_CSV_PATH,
    RAG_EMBEDDING_MODEL,
    RAG_RETRIEVE_MULTIPLIER,
)


def _doc_text(row: pd.Series) -> str:
    parts = [
        str(row.get("title", "") or ""),
        str(row.get("type", "") or ""),
        str(row.get("listed_in", "") or "").replace(",", " "),
        str(row.get("description", "") or "")[:2000],
        str(row.get("director", "") or ""),
        str(row.get("cast", "") or "")[:500],
    ]
    return " \n ".join(p for p in parts if p)


class NetflixRAGIndex:
    """
    Embedding retrieval over Netflix CSV metadata.
    Uses FAISS inner-product on L2-normalized vectors (cosine similarity).
    """

    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._meta: list[dict[str, Any]] = []
        self._dim: Optional[int] = None

    def _cache_dir(self) -> Path:
        d = NETFLIX_CSV_PATH.parent / ".rag_cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _fingerprint(self, csv_path: Path) -> str:
        st = csv_path.stat()
        raw = f"{csv_path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _paths(self, fp: str) -> tuple[Path, Path]:
        base = self._cache_dir()
        return base / f"netflix_{fp}.faiss", base / f"netflix_{fp}.meta.json"

    def ensure_loaded(self) -> None:
        if self._index is not None and self._meta:
            return

        csv_path = NETFLIX_CSV_PATH
        if not csv_path.exists():
            raise FileNotFoundError(f"Netflix catalog not found at {csv_path}")

        fp = self._fingerprint(csv_path)
        faiss_path, meta_path = self._paths(fp)

        if faiss_path.exists() and meta_path.exists():
            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._index = faiss.read_index(str(faiss_path))
            self._dim = self._index.d
            self._get_model()
            return

        df = pd.read_csv(csv_path)
        texts = [_doc_text(df.iloc[i]) for i in range(len(df))]
        self._meta = []
        for i in range(len(df)):
            row = df.iloc[i]
            yr = row.get("release_year")
            self._meta.append(
                {
                    "show_id": str(row.get("show_id", "")),
                    "type": str(row.get("type", "")),
                    "title": str(row.get("title", "")),
                    "release_year": int(yr) if pd.notna(yr) else None,
                    "rating": str(row["rating"]) if pd.notna(row.get("rating")) else None,
                    "listed_in": str(row["listed_in"])
                    if pd.notna(row.get("listed_in"))
                    else None,
                    "description": str(row["description"])[:1200]
                    if pd.notna(row.get("description"))
                    else None,
                    "country": str(row["country"]) if pd.notna(row.get("country")) else None,
                }
            )

        model = self._get_model()
        batch = 64
        embeddings = []
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            emb = model.encode(
                chunk,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > batch,
            )
            embeddings.append(np.asarray(emb, dtype=np.float32))
        mat = np.vstack(embeddings)
        dim = mat.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(mat)
        self._index = index
        self._dim = dim

        faiss.write_index(index, str(faiss_path))
        meta_path.write_text(json.dumps(self._meta), encoding="utf-8")

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(RAG_EMBEDDING_MODEL)
        return self._model

    def search(
        self,
        query: str,
        top_k: int = 8,
        content_type: Optional[str] = None,
        genre_substring: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        self.ensure_loaded()
        assert self._index is not None

        model = self._get_model()
        qv = model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype=np.float32)
        n_probe = min(len(self._meta), max(top_k * RAG_RETRIEVE_MULTIPLIER, top_k * 2))
        scores, indices = self._index.search(qv, n_probe)

        results: list[dict[str, Any]] = []
        ct = content_type.strip() if content_type else None
        gs = genre_substring.strip().lower() if genre_substring else None

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self._meta[int(idx)]
            if ct and row.get("type") != ct:
                continue
            if gs:
                li = (row.get("listed_in") or "").lower()
                if gs not in li:
                    continue
            y = row.get("release_year")
            if year_min is not None and y is not None and y < year_min:
                continue
            if year_max is not None and y is not None and y > year_max:
                continue
            row = {**row, "score": float(score)}
            results.append(row)
            if len(results) >= top_k:
                break

        return results


_rag_singleton: Optional[NetflixRAGIndex] = None


def get_rag() -> NetflixRAGIndex:
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = NetflixRAGIndex()
    return _rag_singleton
