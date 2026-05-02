"""Configurable chat models: OpenAI (GPT), Anthropic (Claude), or Groq — controlled via env."""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


def get_chat_model() -> BaseChatModel:
    """
    LLM_PROVIDER: auto | openai | anthropic | groq
    auto prefers OpenAI → Anthropic → Groq based on available API keys.
    """
    provider = os.getenv("LLM_PROVIDER", "auto").lower().strip()

    def openai() -> BaseChatModel:
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0.65)

    def anthropic_model() -> BaseChatModel:
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        return ChatAnthropic(model=model, temperature=0.65)

    def groq() -> BaseChatModel:
        model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        return ChatGroq(model=model, temperature=0.65)

    if provider == "openai":
        return openai()
    if provider == "anthropic":
        return anthropic_model()
    if provider == "groq":
        return groq()

    if provider != "auto":
        raise ValueError(
            "LLM_PROVIDER must be one of: auto, openai, anthropic, groq"
        )

    if os.getenv("OPENAI_API_KEY"):
        return openai()
    if os.getenv("ANTHROPIC_API_KEY"):
        return anthropic_model()
    if os.getenv("GROQ_API_KEY"):
        return groq()

    raise RuntimeError(
        "No LLM API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY "
        "(optionally force provider with LLM_PROVIDER)."
    )
