from fastapi import APIRouter

from backend.schemas import ChatRequest, ChatResponse
from backend.services.agent_service import chat_turn

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def post_chat(body: ChatRequest) -> ChatResponse:
    """LLM + LangGraph + RAG — decoupled from `/titles` structured filtering."""
    sid, reply = chat_turn(body.session_id or "", body.message)
    return ChatResponse(session_id=sid, reply=reply)
