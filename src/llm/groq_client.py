from __future__ import annotations

from typing import Any


def create_groq_client(api_key: str | None = None) -> Any:
    from src.config import settings

    key = (api_key or settings.GROQ_API_KEY).strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to the project-root .env file.")

    from groq import Groq

    return Groq(api_key=key)