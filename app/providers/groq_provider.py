"""
app/providers/groq_provider.py

Groq as the confidence-based fallback provider.

Groq runs open-source models (Llama 4, Llama 3.2 Vision) on custom LPU
hardware — fastest inference latency of any commercial API. Free tier
with generous rate limits, no credit card required.

Groq exposes an OpenAI-compatible API, so the call format is:
  POST https://api.groq.com/openai/v1/chat/completions
  Authorization: Bearer {api_key}

Image format: base64 data URI in the image_url content block —
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}

Get a free API key at: console.groq.com
Model list: console.groq.com/docs/models

Recommended vision models (as of 2026):
  meta-llama/llama-4-scout-17b-16e-instruct  — best balance, default
  meta-llama/llama-4-maverick-17b-128e-instruct — more capable, slower
  llama-3.2-11b-vision-preview               — smaller, very fast
  llama-3.2-90b-vision-preview               — largest, most accurate
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import httpx

from app.config import get_settings
from app.models.po_models import ModelSource
from app.providers.base import (
    BaseProvider,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

_GROQ_API_BASE = "https://api.groq.com/openai/v1"


class GroqProvider(BaseProvider):
    """
    Extracts PO fields using Groq's vision models as a fallback.

    Drop-in replacement for ClaudeProvider — implements the same three
    abstract members (model_name, source, extract_raw) so router_service
    doesn't need any changes beyond swapping the provider instance.

    Uses the OpenAI-compatible chat completions endpoint.
    No retry logic — Groq's API is highly available and fast enough
    that a failed request should be surfaced immediately.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._settings.groq_model

    @property
    def source(self) -> ModelSource:
        return ModelSource.GROQ

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    async def extract_raw(
        self,
        prompt: str,
        image_bytes: Optional[bytes],
        image_mime_type: Optional[str],
    ) -> tuple[str, Optional[int], Optional[int]]:
        """
        POST to Groq /v1/chat/completions and return
        (response_text, prompt_tokens, completion_tokens).

        Image handling:
          Groq vision models accept base64 images as data URIs in the
          image_url content block — same format as OpenAI GPT-4V.
          Full resolution images are sent (Groq is cloud-side, no VRAM limit).

        Content block ordering: image first, then text prompt.
        Same rationale as Claude — model processes the image before
        reading the extraction instructions.
        """
        if not self._settings.groq_api_key:
            raise ProviderUnavailableError(
                self.model_name,
                "GROQ_API_KEY is not set in .env"
            )

        url = f"{_GROQ_API_BASE}/chat/completions"
        headers = self._build_headers()
        messages = self._build_messages(prompt, image_bytes, image_mime_type)

        payload = {
            "model":       self.model_name,
            "messages":    messages,
            "temperature": 0,           # deterministic extraction
            "max_tokens":  self._settings.claude_max_tokens,  # reuse Claude's limit
        }

        timeout = self._settings.claude_timeout_seconds

        logger.info(
            "Groq fallback call: model=%s, has_image=%s, timeout=%ds",
            self.model_name,
            image_bytes is not None,
            timeout,
        )

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)

        except httpx.ConnectError as e:
            raise ProviderUnavailableError(
                self.model_name,
                f"Cannot reach Groq API: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                self.model_name,
                f"Groq did not respond within {timeout}s: {e}"
            ) from e

        # Handle HTTP errors with actionable messages
        if response.status_code == 401:
            raise ProviderResponseError(
                self.model_name, 401,
                "Invalid API key. Check GROQ_API_KEY in .env. "
                "Get a key at console.groq.com"
            )
        if response.status_code == 429:
            raise ProviderResponseError(
                self.model_name, 429,
                "Groq rate limit hit. Free tier: 30 req/min. "
                "Reduce CONFIDENCE_THRESHOLD to call fallback less often."
            )
        if response.status_code == 400:
            raise ProviderResponseError(
                self.model_name, 400,
                f"Bad request — model may not support vision: {response.text[:300]}"
            )
        if response.status_code != 200:
            raise ProviderResponseError(
                self.model_name,
                response.status_code,
                response.text[:300],
            )

        data = response.json()

        # Extract text from OpenAI-format response
        try:
            response_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderResponseError(
                self.model_name, 200,
                f"Unexpected response shape: {e} — {str(data)[:300]}"
            )

        if not response_text:
            raise ProviderResponseError(
                self.model_name, 200,
                f"Empty response content. Finish reason: "
                f"{data['choices'][0].get('finish_reason')}"
            )

        usage = data.get("usage", {})
        prompt_tokens     = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        logger.info(
            "Groq response received. prompt_tokens=%s completion_tokens=%s "
            "finish_reason=%s",
            prompt_tokens,
            completion_tokens,
            data["choices"][0].get("finish_reason"),
        )

        # Warn on truncated response
        if data["choices"][0].get("finish_reason") == "length":
            logger.warning(
                "Groq response hit max_tokens (%d). "
                "Some fields may be missing. "
                "Increase CLAUDE_MAX_TOKENS in .env.",
                self._settings.claude_max_tokens,
            )

        return response_text, prompt_tokens, completion_tokens

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Return True if Groq API is reachable and the configured model is available.
        Uses GET /v1/models — free, no token generation.
        """
        if not self._settings.groq_api_key:
            logger.warning(
                "Groq health check skipped: GROQ_API_KEY not set in .env"
            )
            return False

        url = f"{_GROQ_API_BASE}/models"
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(url, headers=self._build_headers())

            if response.status_code == 401:
                logger.error(
                    "Groq health check failed: invalid API key. "
                    "Check GROQ_API_KEY in .env."
                )
                return False

            if response.status_code != 200:
                logger.error(
                    "Groq health check failed: HTTP %d", response.status_code
                )
                return False

            data = response.json()
            available_ids = [m.get("id", "") for m in data.get("data", [])]

            if self.model_name not in available_ids:
                logger.warning(
                    "Groq model '%s' not found. Available models: %s. "
                    "Check GROQ_MODEL in .env.",
                    self.model_name,
                    available_ids[:8],
                )
                return False

            logger.debug(
                "Groq health check passed. Model '%s' available.", self.model_name
            )
            return True

        except httpx.ConnectError:
            logger.error("Groq health check failed: cannot reach %s", _GROQ_API_BASE)
            return False
        except httpx.TimeoutException:
            logger.error("Groq health check timed out.")
            return False
        except Exception as e:
            logger.error("Groq health check unexpected error: %s", e)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict:
        """Groq uses standard Bearer auth — OpenAI-compatible format."""
        return {
            "Authorization": f"Bearer {self._settings.groq_api_key}",
            "Content-Type":  "application/json",
        }

    @staticmethod
    def _build_messages(
        prompt: str,
        image_bytes: Optional[bytes],
        image_mime_type: Optional[str],
    ) -> list:
        """
        Build the OpenAI-format messages array.

        With image:
          [{"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
              {"type": "text", "text": "..."}
          ]}]

        Without image (text-only models or no image provided):
          [{"role": "user", "content": "..."}]

        Image before text — model sees the document before reading instructions.
        """
        if image_bytes is None:
            # Text-only request — simple string content
            return [{"role": "user", "content": prompt}]

        mime = image_mime_type or "image/jpeg"
        b64  = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime};base64,{b64}"

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]