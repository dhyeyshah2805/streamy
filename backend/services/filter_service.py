from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from backend.config import NETFLIX_CSV_PATH
from backend.schemas import TitleFilterParams, TitleOut


def _norm(s: str | float | None) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip().lower()


@lru_cache(maxsize=1)
def _load_df() -> pd.DataFrame:
    path = NETFLIX_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Netflix catalog not found at {path}")
    df = pd.read_csv(path)
    # Normalize column names we rely on
    for col in ("title", "description", "listed_in", "type", "country", "rating"):
        if col not in df.columns:
            df[col] = ""
    df["release_year"] = pd.to_numeric(df.get("release_year"), errors="coerce")
    return df


def query_titles(params: TitleFilterParams) -> tuple[int, list[TitleOut]]:
    """Structured filtering over the Netflix CSV — fast pandas paths, no LLM."""
    df = _load_df().copy()
    mask = pd.Series(True, index=df.index)

    if params.type:
        mask &= df["type"].astype(str).str.strip() == params.type

    if params.country:
        sub = _norm(params.country)
        mask &= df["country"].fillna("").astype(str).str.lower().str.contains(
            re.escape(sub), regex=True, na=False
        )

    if params.genre:
        sub = _norm(params.genre)
        mask &= df["listed_in"].fillna("").astype(str).str.lower().str.contains(
            re.escape(sub), regex=True, na=False
        )

    if params.year_min is not None:
        mask &= df["release_year"] >= params.year_min
    if params.year_max is not None:
        mask &= df["release_year"] <= params.year_max

    if params.q:
        q = params.q.strip().lower()
        desc = df["description"].fillna("").astype(str).str.lower()
        title = df["title"].fillna("").astype(str).str.lower()
        mask &= title.str.contains(re.escape(q), regex=True, na=False) | desc.str.contains(
            re.escape(q), regex=True, na=False
        )

    filtered = df.loc[mask]
    total = int(len(filtered))
    page = filtered.iloc[params.offset : params.offset + params.limit]

    items: list[TitleOut] = []
    for _, row in page.iterrows():
        yr = row.get("release_year")
        raw_desc = row.get("description")
        if pd.isna(raw_desc) or raw_desc is None:
            desc_out = None
        else:
            s = str(raw_desc)
            desc_out = s[:500] + "..." if len(s) > 500 else s

        items.append(
            TitleOut(
                show_id=str(row.get("show_id", "")),
                type=str(row.get("type", "")),
                title=str(row.get("title", "")),
                release_year=int(yr) if pd.notna(yr) else None,
                rating=str(row["rating"]) if pd.notna(row.get("rating")) else None,
                listed_in=str(row["listed_in"]) if pd.notna(row.get("listed_in")) else None,
                description=desc_out,
            )
        )
    return total, items


def invalidate_catalog_cache() -> None:
    _load_df.cache_clear()
