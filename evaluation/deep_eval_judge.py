"""Custom Judge implementation for DeepEval using Groq models.

Inherits from DeepEvalBaseLLM and handles:
- Structured output (Pydantic schema validation)
- Reasoning model output sanitization (stripping <think>...</think> tags from Qwen models)
- JSON extraction from markdown code fences
- Rate limit (HTTP 429) backoff and retries
- Fallback / provider compatibility handling
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional, Type
from pydantic import BaseModel

THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_clean_json(text: str) -> str:
    """Removes thinking tags and extracts valid JSON substring."""
    # 1. Strip <think>...</think> tags if emitted by reasoning models like Qwen 3.8
    cleaned = THINK_TAG_PATTERN.sub("", text).strip()

    # 2. Extract from markdown code fence if present
    match = JSON_FENCE_PATTERN.search(cleaned)
    if match:
        return match.group(1).strip()

    # 3. Find outermost braces
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1].strip()

    return cleaned


try:
    from deepeval.models import DeepEvalBaseLLM
except ImportError:
    # If deepeval is not yet installed in local environment, provide a compatible base
    class DeepEvalBaseLLM:  # type: ignore[no-redef]
        def __init__(self, model: str | None = None, *args: Any, **kwargs: Any) -> None:
            self.model_name = model or kwargs.get("model_name", "")
            self.name = self.model_name
            self.model = self.load_model() if hasattr(self, "load_model") else None

        def load_model(self) -> Any:
            raise NotImplementedError

        def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None, *args: Any, **kwargs: Any) -> str | BaseModel:
            raise NotImplementedError

        async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None, *args: Any, **kwargs: Any) -> str | BaseModel:
            raise NotImplementedError

        def get_model_name(self, *args: Any, **kwargs: Any) -> str:
            return self.model_name


class GroqJudge(DeepEvalBaseLLM):
    """Custom LLM-as-a-Judge for DeepEval powered by Groq."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 4,
        rate_limit_delay_seconds: float = 0.5,
    ) -> None:
        self.model_name = model_name or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        self.api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self.temperature = temperature
        self.max_retries = max_retries
        self.rate_limit_delay_seconds = rate_limit_delay_seconds
        self._client: Any = None
        # Call super().__init__ with model parameter required by DeepEvalBaseLLM
        super().__init__(model=self.model_name)

    def load_model(self) -> Any:
        if self._client is None:
            from groq import Groq

            if not self.api_key:
                raise ValueError("GROQ_API_KEY is not set. Provide api_key or set GROQ_API_KEY environment variable.")
            self._client = Groq(api_key=self.api_key)
        return self._client

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return f"Groq-{self.model_name}"

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str | BaseModel:
        client = self.load_model()

        if self.rate_limit_delay_seconds > 0:
            time.sleep(self.rate_limit_delay_seconds)

        # Prepare messages
        system_content = "You are an objective and precise evaluation judge for a mutual fund RAG application."
        user_content = prompt

        if schema is not None:
            system_content += (
                " You must output ONLY a valid JSON object matching the requested schema. "
                "Do NOT output markdown fences, thinking tokens, or additional text."
            )
            try:
                schema_json = json.dumps(schema.model_json_schema(), indent=2)
                user_content += f"\n\nOutput MUST strictly follow this JSON schema:\n{schema_json}"
            except Exception:
                pass

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Groq Chat Completion with json_object format if schema provided
                req_kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": 1024,
                }
                if schema is not None:
                    req_kwargs["response_format"] = {"type": "json_object"}

                response = client.chat.completions.create(**req_kwargs)
                raw_text = response.choices[0].message.content or ""

                if schema is not None:
                    clean_json = extract_clean_json(raw_text)
                    return schema.model_validate_json(clean_json)

                return raw_text.strip()
            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                # If rate limited (HTTP 429), back off exponentially
                if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
                    wait_time = 2.0 * (2**attempt)
                    match = re.search(r"try again in ([0-9.]+)s", err_str)
                    if match:
                        wait_time = max(wait_time, float(match.group(1)) + 0.5)
                    time.sleep(wait_time)
                else:
                    time.sleep(1.0)

        # If retries exhausted and schema validation failed, raise informative error
        raise RuntimeError(f"GroqJudge failed after {self.max_retries} attempts: {last_error}")

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str | BaseModel:
        # DeepEval async runner falls back to synchronous generate
        return self.generate(prompt, schema, *args, **kwargs)


class OpenAIJudge(DeepEvalBaseLLM):
    """OpenAI evaluation judge for comparison or fallback when configured."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        self._client: Any = None
        super().__init__(model=self.model_name)

    def load_model(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return f"OpenAI-{self.model_name}"

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str | BaseModel:
        client = self.load_model()
        messages = [
            {"role": "system", "content": "You are a precise evaluation judge for a RAG system."},
            {"role": "user", "content": prompt},
        ]
        if schema is not None:
            completion = client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=schema,
            )
            return completion.choices[0].message.parsed
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return completion.choices[0].message.content or ""

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str | BaseModel:
        return self.generate(prompt, schema, *args, **kwargs)

