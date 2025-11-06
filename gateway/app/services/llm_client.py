from __future__ import annotations
from typing import Dict, Any
import asyncio
import httpx
import re
from datetime import datetime

from app.core.settings import Settings
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChoiceMessage,
    Usage,
)

import time
from uuid import uuid4

def _gen_chat_id() -> str:
    return f"chatcmpl-{uuid4().hex[:12]}"

def _safe_created(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return int(time.time())

# ---- Custom exceptions for precise error mapping ----
class UpstreamTimeout(Exception):
    pass


class UpstreamHTTPError(Exception):
    def __init__(self, status_code: int, body: Any | None = None):
        super().__init__(f"Upstream HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


_OLLAMA_SCHEME = re.compile(r"^ollama://", re.IGNORECASE)


def _to_http_url(url: str) -> str:
    # ollama://host:port  -> http://host:port
    return _OLLAMA_SCHEME.sub("http://", url, count=1)


class LLMClient:
    """
    Async client for:
      • vLLM (OpenAI-compatible) at {base}/v1/chat/completions
      • Ollama (native) at {base}/api/chat           [auto when base starts with ollama://]

    Non-streaming only (Day 2).
    """

    def __init__(self, settings: Settings):
        self.raw_base_url = settings.VLLM_BASE_URL.rstrip("/")
        self.is_ollama = bool(_OLLAMA_SCHEME.match(self.raw_base_url))
        self.base_url = _to_http_url(self.raw_base_url) if self.is_ollama else self.raw_base_url

        self.connect_timeout = settings.HTTP_CONNECT_TIMEOUT
        self.read_timeout = settings.HTTP_READ_TIMEOUT
        self.write_timeout = settings.HTTP_WRITE_TIMEOUT
        self.total_timeout = settings.HTTP_TOTAL_TIMEOUT
        self.retry_connect_errors = settings.RETRY_CONNECT_ERRORS

        self._timeout = httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.write_timeout,
            pool=self.total_timeout,
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
            headers={"Content-Type": "application/json"},
        )

    async def aclose(self):
        await self._client.aclose()

    async def chat_completion(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        if self.is_ollama:
            return await self._ollama_chat_completion(req)
        return await self._vllm_chat_completion(req)

    # ---------- vLLM (OpenAI-compatible) ----------
    async def _vllm_chat_completion(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        body: Dict[str, Any] = {
            "model": req.model,
            "messages": [m.model_dump() for m in req.messages],
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            "stream": False,  # Day 2: non-stream
        }
        if req.seed is not None:
            body["seed"] = req.seed
        if req.user is not None:
            body["user"] = req.user

        return await self._post_openai_style("/v1/chat/completions", body, req)

    # ---------- Ollama (native /api/chat) ----------
    async def _ollama_chat_completion(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Maps our OpenAI-style request to Ollama /api/chat and then back to OpenAI-style response.
        """
        # 1) Build options (Ollama-specific)
        options: Dict[str, Any] = {}

        if req.temperature is not None:
            options["temperature"] = req.temperature
        if req.top_p is not None:
            options["top_p"] = req.top_p

        # Accept both max_tokens and legacy max_token
        max_tokens = getattr(req, "max_tokens", None)
        if max_tokens is None:
            max_tokens = getattr(req, "max_token", None)
        if max_tokens is not None:
            # Ollama uses num_predict for max new tokens
            options["num_predict"] = max_tokens

        if req.seed is not None:
            options["seed"] = req.seed

        # 2) Build the Ollama body once (no need to set body["max_tokens"] here)
        body: Dict[str, Any] = {
            "model": req.model,
            "messages": [m.model_dump() for m in req.messages],
            "stream": False,  # Day 2: non-stream
            "options": options,
        }

        # 3) POST to Ollama
        attempt = 0
        while True:
            try:
                r = await self._client.post("/api/chat", json=body)
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectError) as e:
                attempt += 1
                if attempt > self.retry_connect_errors:
                    raise UpstreamTimeout(f"Ollama timeout/connect error after {attempt} attempts: {e}") from e
                await asyncio.sleep(0.2 * attempt)
                continue

            if r.status_code >= 500:
                raise UpstreamHTTPError(r.status_code, _safe_json(r))
            if r.status_code >= 400:
                raise UpstreamHTTPError(r.status_code, _safe_json(r))

            data = r.json()
            # Example (non-stream):
            # {
            #   "model": "...",
            #   "created_at": "2025-11-06T...",
            #   "message": {"role":"assistant","content":"..."},
            #   "prompt_eval_count": 23,
            #   "eval_count": 54,
            #   "done": true
            # }

            content = ""
            msg = data.get("message") or {}
            if isinstance(msg, dict):
                content = msg.get("content", "") or ""

            prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
            completion_tokens = int(data.get("eval_count", 0) or 0)

            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            created_ts = _iso8601_to_unix(data.get("created_at"))
            return ChatCompletionResponse(
                id=_gen_chat_id(),
                object="chat.completion",
                created=created_ts or 0,
                model=data.get("model", req.model),
                choices=[ChatChoice(index=0, message=ChoiceMessage(content=content), finish_reason="stop")],
                usage=usage,
            )

    # ---------- shared POST helper for OpenAI-style upstreams ----------
    async def _post_openai_style(self, path: str, body: Dict[str, Any], req: ChatCompletionRequest) -> ChatCompletionResponse:
        attempt = 0
        while True:
            try:
                r = await self._client.post(path, json=body)
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectError) as e:
                attempt += 1
                if attempt > self.retry_connect_errors:
                    raise UpstreamTimeout(f"Timeout/connect error after {attempt} attempts: {e}") from e
                await asyncio.sleep(0.2 * attempt)
                continue

            if r.status_code >= 500:
                raise UpstreamHTTPError(r.status_code, _safe_json(r))
            if r.status_code >= 400:
                raise UpstreamHTTPError(r.status_code, _safe_json(r))

            data = r.json()

            # Ensure id is a non-empty string
            resp_id = data.get("id")
            if not isinstance(resp_id, str) or not resp_id:
                resp_id = _gen_chat_id()

            # Ensure created is an int (fallback to now)
            created_val = _safe_created(data.get("created"))

            # Normalize choices
            raw_choices = data.get("choices") or []
            if not isinstance(raw_choices, list) or not raw_choices:
                raw_choices = [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]

            choices: list[ChatChoice] = []
            for i, c in enumerate(raw_choices):
                msg = c.get("message") or {"role": "assistant", "content": ""}
                # Guard against malformed message
                role = msg.get("role", "assistant") or "assistant"
                content = msg.get("content", "") or ""
                choices.append(
                    ChatChoice(
                        index=c.get("index", i),
                        message=ChoiceMessage(role=role, content=content),
                        finish_reason=c.get("finish_reason") or "stop",
                    )
                )

            # Normalize usage
            usage_json = data.get("usage") or {}
            prompt_tokens = int(usage_json.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage_json.get("completion_tokens", 0) or 0)
            total_tokens = int(usage_json.get("total_tokens", prompt_tokens + completion_tokens) or 0)

            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            return ChatCompletionResponse(
                id=resp_id,
                object=data.get("object", "chat.completion"),
                created=created_val,
                model=data.get("model", body.get("model", req.model)),
                choices=choices,
                usage=usage,
            )


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"text": resp.text[:1000]}


def _iso8601_to_unix(iso_str: Any) -> int | None:
    if not iso_str or not isinstance(iso_str, str):
        return None
    try:
        # '2025-11-06T15:10:11.123Z' -> seconds
        return int(datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None