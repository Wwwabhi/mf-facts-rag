from __future__ import annotations

import re
from typing import Any


class GroqRequestError(RuntimeError):
    """Safe, user-facing failure from Groq configuration or request handling."""


SECRET_PATTERN = re.compile(r"\b(?:gsk_|sk-)[A-Za-z0-9_-]{12,}\b")
BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+\S+")


def format_groq_error(exc: Exception, api_key: str = "", model_name: str = "") -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code in {401, 403}:
        return f"Groq rejected the configured API key (HTTP {status_code}). Check GROQ_API_KEY."
    if status_code == 404:
        model = f" '{model_name}'" if model_name else ""
        return f"Groq could not find or use model{model} (HTTP 404). Check GROQ_MODEL."
    if status_code == 429:
        return "Groq rate limit or quota exceeded (HTTP 429). Check the account quota and try again later."
    if isinstance(exc, (TimeoutError, ConnectionError)) or "connection" in type(exc).__name__.casefold():
        return "Could not connect to Groq. Check the network connection and try again."

    detail = str(exc).strip()
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    detail = BEARER_PATTERN.sub("Bearer [REDACTED]", detail)
    detail = SECRET_PATTERN.sub("[REDACTED]", detail)
    if status_code is not None:
        return f"Groq request failed (HTTP {status_code}): {detail or type(exc).__name__}"
    if detail:
        return f"Groq request failed: {detail}"
    return f"Groq request failed: {type(exc).__name__}"


def create_groq_client(api_key: str | None = None) -> Any:
    from src.config import settings

    key = (api_key or settings.GROQ_API_KEY).strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to the project-root .env file.")

    from groq import Groq

    return Groq(api_key=key)