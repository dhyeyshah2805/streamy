from typing import Literal, Optional

from fastapi import APIRouter, Query

from backend.schemas import TitleFilterParams, TitleListResponse
from backend.services.filter_service import query_titles

router = APIRouter(tags=["catalog"])


@router.get("/titles", response_model=TitleListResponse)
def list_titles(
    q: Optional[str] = Query(None, description="Substring match on title or description"),
    type: Optional[Literal["Movie", "TV Show"]] = Query(None, alias="type"),
    genre: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    year_min: Optional[int] = Query(None),
    year_max: Optional[int] = Query(None),
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> TitleListResponse:
    """Structured catalog filtering — no LLM (fast path for discovery UI)."""
    params = TitleFilterParams(
        q=q,
        type=type,
        genre=genre,
        country=country,
        year_min=year_min,
        year_max=year_max,
        limit=limit,
        offset=offset,
    )
    total, items = query_titles(params)
    return TitleListResponse(total=total, items=items)
