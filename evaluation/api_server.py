"""Lightweight HTTP API server wrapper for the mutual fund RAG pipeline.

Useful when evaluating the chatbot over HTTP or deploying as a standalone REST API microservice.

Run:
    uvicorn evaluation.api_server:app --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    # FastAPI is optional; provide message if launched without it
    FastAPI = None  # type: ignore[assignment,misc]

if FastAPI is not None:
    app = FastAPI(
        title="Mutual Fund Facts RAG API",
        version="0.1.0",
        description="REST API endpoint for factual mutual fund questions.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ChatRequest(BaseModel):
        question: str = Field(..., description="Mutual fund question", example="What is the exit load for HDFC Flexi Cap Fund?")
        top_k: int = Field(5, description="Number of chunks to retrieve", ge=1, le=20)

    class ChatResponse(BaseModel):
        answer: str
        source_url: str | None = None
        source_date: str | None = None
        category: str
        allowed: bool
        retrieval_context: list[str] = Field(default_factory=list)

    @app.get("/healthz")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat_endpoint(payload: ChatRequest) -> ChatResponse:
        from src.retrieval.query_pipeline import answer_question

        try:
            res = answer_question(payload.question, top_k=payload.top_k)
            contexts = [
                f"{c.get('metadata', {}).get('scheme', '')}: {c.get('content', '')}".strip()
                for c in res.retrieved_chunks
                if c.get("content")
            ]
            return ChatResponse(
                answer=res.answer or res.decision.message or "",
                source_url=res.source_url,
                source_date=res.source_date,
                category=res.decision.category,
                allowed=res.decision.allowed,
                retrieval_context=contexts,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
else:
    app = None  # type: ignore[assignment]
