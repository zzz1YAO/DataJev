"""Thin OpenAI-compatible JSON client shared by the LLM analyst and the LLM
baseline controller.

Kept deliberately small: one call shape (system + user -> JSON object) and one
usage counter. No framework.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class LLMError(RuntimeError):
    """Raised when the LLM backend is missing, unreachable, or unparseable."""


@dataclass
class LLMConfig:
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_s: float = 120.0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def describe(self) -> str:
        return f"{self.model} @ {self.base_url or 'default OpenAI endpoint'}"


@dataclass
class LLMResponse:
    data: dict[str, Any]
    text: str
    usage: dict[str, Any] | None = None
    parse_error: str | None = None
    finish_reason: str | None = None


@dataclass
class LLMClient:
    """Synchronous JSON-only chat client."""

    config: LLMConfig
    total_usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "total_tokens": 0}
    )

    @property
    def available(self) -> bool:
        return self.config.available

    def json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if not self.config.available:
            raise LLMError(
                "No LLM API key configured. Set DATAJEV_LLM_API_KEY, OPENAI_API_KEY, "
                "DEEPSEEK_API_KEY or deepseek_api."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError("The 'openai' package is required for the LLM backend.") from exc

        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs
            )
        except Exception as exc:  # pragma: no cover - backend specific
            if "response_format" not in str(exc):
                raise LLMError(f"LLM call failed: {exc}") from exc
            response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = _usage_dict(response)
        self._accumulate(usage)
        data, parse_error = _parse_json(text)
        return LLMResponse(
            data=data,
            text=text,
            usage=usage,
            parse_error=parse_error,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    def _accumulate(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            self.total_usage["calls"] += 1
            return
        self.total_usage["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.total_usage[key] += int(usage.get(key) or 0)


def _usage_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse_json(text: str) -> tuple[dict[str, Any], str | None]:
    """Best-effort extraction of one JSON object plus a diagnostic on failure."""
    candidate = text.strip()
    if not candidate:
        return {}, "empty response"
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
        return (parsed if isinstance(parsed, dict) else {"value": parsed}), None
    except json.JSONDecodeError as exc:
        first_error = exc
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return (parsed if isinstance(parsed, dict) else {"value": parsed}), None
        except json.JSONDecodeError as exc:
            first_error = exc
    return {"_unparsed": text}, f"invalid JSON ({first_error.msg} at character {first_error.pos})"


def resolve_llm_config(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMConfig:
    """Resolve an OpenAI-compatible backend from explicit args then env vars."""
    key = api_key or os.environ.get("DATAJEV_LLM_API_KEY")
    resolved_base = base_url or os.environ.get("DATAJEV_LLM_BASE_URL")

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("deepseek_api")
    if not key and deepseek_key:
        key = deepseek_key
        resolved_base = resolved_base or DEFAULT_DEEPSEEK_BASE_URL
        default_model = "deepseek-flash"
    else:
        if not key:
            key = os.environ.get("OPENAI_API_KEY")
        resolved_base = resolved_base or os.environ.get("OPENAI_BASE_URL")
        default_model = "gpt-4o-mini"

    resolved_model = model or os.environ.get("DATAJEV_LLM_MODEL") or default_model
    return LLMConfig(model=resolved_model, api_key=key, base_url=resolved_base)
