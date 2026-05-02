from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = Field(
        None,
        description="Stable id for conversational memory (LangGraph thread).",
    )


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class TitleOut(BaseModel):
    show_id: str
    type: str
    title: str
    release_year: Optional[int] = None
    rating: Optional[str] = None
    listed_in: Optional[str] = None
    description: Optional[str] = None


class TitleFilterParams(BaseModel):
    """API-driven filters — no LLM. Applied as structured query over the catalog."""

    q: Optional[str] = Field(None, description="Substring match on title or description")
    type: Optional[Literal["Movie", "TV Show"]] = None
    genre: Optional[str] = Field(None, description="Substring match on listed_in / genres")
    country: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    limit: int = Field(25, ge=1, le=200)
    offset: int = Field(0, ge=0)


class TitleListResponse(BaseModel):
    total: int
    items: list[TitleOut]
