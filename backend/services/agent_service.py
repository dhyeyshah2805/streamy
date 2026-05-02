"""LangGraph agent: LLM intent + tools + MemorySaver for session-scoped conversational context."""

from __future__ import annotations

import os
import uuid
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from backend.config import RAG_TOP_K_DEFAULT
from backend.services.llm_factory import get_chat_model
from backend.services.rag_service import get_rag

SYSTEM_PROMPT = """You are a Netflix discovery assistant with access to a retrieval tool over a Netflix catalog.

Use the user's message and prior conversation in this thread to infer tastes (genres, eras, mood, movie vs TV).
When the user asks for recommendations, call search_netflix_catalog with a focused semantic query (not raw chit-chat).
After tool results, synthesize a short, helpful answer with a few concrete titles and why they fit.

If retrieval returns nothing relevant, say so honestly and suggest refining the query or filters.
"""


def _format_hits(rows: list[dict]) -> str:
    lines = []
    for i, r in enumerate(rows, 1):
        title = r.get("title", "?")
        ctype = r.get("type", "?")
        year = r.get("release_year", "")
        genres = r.get("listed_in") or ""
        desc = (r.get("description") or "")[:280]
        score = r.get("score", 0)
        lines.append(
            f"{i}. {title} ({ctype}, {year}) [sim={score:.3f}]\n"
            f"   Genres/tags: {genres}\n"
            f"   {desc}"
        )
    return "\n\n".join(lines)


@tool
def search_netflix_catalog(
    query: str,
    content_type: Optional[str] = None,
    genre_hint: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> str:
    """Semantic search over Netflix title metadata (RAG). Use a rich query (themes, tone, similar-to, actors).
    content_type: 'Movie' or 'TV Show' when the user clearly wants one medium.
    genre_hint: substring such as 'thriller' or 'Stand-Up' to narrow listed_in / genres.
    """
    rag = get_rag()
    ct: str | None = None
    if content_type:
        c = content_type.strip()
        if c in ("Movie", "TV Show"):
            ct = c

    top_k = int(os.getenv("RAG_TOP_K", str(RAG_TOP_K_DEFAULT)))
    rows = rag.search(
        query,
        top_k=top_k,
        content_type=ct,
        genre_substring=genre_hint,
        year_min=year_min,
        year_max=year_max,
    )
    if not rows:
        return "No matching titles found for that query. Try alternate wording or relax filters."
    return _format_hits(rows)


def _agent_node(state: MessagesState) -> dict[str, list[BaseMessage]]:
    llm = get_chat_model().bind_tools([search_netflix_catalog])
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm.invoke(msgs)
    return {"messages": [response]}


def build_agent():
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", ToolNode([search_netflix_catalog]))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def chat_turn(session_id: str, user_text: str) -> tuple[str, str]:
    """Returns (session_id, assistant_reply_text)."""
    sid = (session_id or "").strip() or str(uuid.uuid4())
    agent = get_agent()
    config = {"configurable": {"thread_id": sid}}
    out = agent.invoke(
        {"messages": [HumanMessage(content=user_text)]},
        config,
    )
    messages = out["messages"]
    last = messages[-1]
    if isinstance(last, AIMessage):
        text = last.content if isinstance(last.content, str) else str(last.content)
    else:
        text = str(last.content)
    return sid, text
