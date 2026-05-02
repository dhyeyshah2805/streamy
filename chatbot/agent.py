"""
CLI for the same LangGraph + RAG stack as `backend.main` / `POST /api/chat`.

Run from the repository root so imports resolve:

  uv run python chatbot/agent.py

Requires OPENAI_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY (see backend/services/llm_factory.py).
"""

from __future__ import annotations

import os
import uuid

from backend.services.agent_service import chat_turn
from backend.services.rag_service import get_rag


def start_chat() -> None:
    session_id = str(uuid.uuid4())
    print("Netflix AI Agent (LangGraph + RAG + session memory)")
    print("Session id:", session_id)
    print("Commands: quit | exit | q")
    print("-" * 48)

    try:
        get_rag().ensure_loaded()
        print("Catalog embeddings ready.\n")
    except Exception as e:
        print(f"Warning: could not load RAG index ({e}). First query may be slow.\n")

    history: list[tuple[str, str]] = []

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        if not user_input:
            continue

        try:
            session_id, reply = chat_turn(session_id, user_input)
            print(f"\nAssistant:\n{reply}")
            history.append((user_input, reply))
            if len(history) > 12:
                history = history[-12:]
        except Exception as e:
            print(f"\nError: {e}")
            if not (
                os.getenv("OPENAI_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("GROQ_API_KEY")
            ):
                print(
                    "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY in your environment."
                )


if __name__ == "__main__":
    start_chat()
