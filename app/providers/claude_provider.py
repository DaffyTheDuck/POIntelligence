"""
app/providers/claude_provider.py

Concrete provider for Claude via the Anthropic Messages API.
Called only when the local Ollama provider scores below the confidence
threshold on one or more fields (architecture decision #3).

Implements the three abstract members from BaseProvider:
  - model_name  — from settings (e.g. "claude-opus-4-5")
  - source      — ModelSource.CLAUDE
  - extract_raw — POST to Anthropic /v1/messages, return raw response text

Key differences from OllamaProvider:
  - Anthropic content block format (typed array vs flat images list)
  - No image resizing — Claude runs on Anthropic infra, full resolution matters
    for the degraded documents that caused local model to fail in the first place
  - Health check via GET /v1/models — validates API key, not just connectivity
  - System prompt injected separately via the API's top-level 'system' field
  - No retry logic — Anthropic's API is highly available; transient failures
    here mean something is genuinely wrong, not a cold-start delay

Cost awareness:
  Claude is called with input.target_fields set to only the low-confidence fields.
  The base class _build_prompt() builds a prompt for just those fields.
  For a typical fallback of 3 fields out of 25, this is ~400 tokens vs ~2000
  for a full document pass — a 5× cost reduction on the fallback path.
"""

from __future__ import annotations

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

# Anthropic API constants
_ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
_ANTHROPIC_VERSION = "2023-06-01"

# System prompt injected via the API's top-level 'system' parameter.
# Kept short — the detailed field instructions live in the user message
# (built by BaseProvider._build_prompt). This just sets Claude's role
# and output contract so it doesn't add preamble text around the JSON.
_SYSTEM_PROMPT = (
    "You are a precise Purchase Order data extraction engine. "
    "You receive OCR-extracted text from purchase order documents and extract "
    "structured field values with confidence scores. "
    "You always respond with valid JSON only — no markdown, no explanation, "
    "no preamble. If a field is not present, use null. Never hallucinate values."
)


class ClaudeProvider(BaseProvider):
    """
    Extracts PO fields using Claude via the Anthropic Messages API.

    Acts as the confidence-based fallback (architecture decision #3):
      - Router calls this when local model confidence < CONFIDENCE_THRESHOLD
      - Only re-extracts the low-confidence fields (target_fields is set)
      - Result merged back into ExtractionResult by extraction_service

    Also acts as the disagreement resolver (architecture decision #4):
      - If local and Claude disagree on a field value (delta > DISAGREEMENT_THRESHOLD)
      - The field is flagged for human review — neither value is trusted blindly
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._settings.claude_model

    @property
    def source(self) -> ModelSource:
        return ModelSource.CLAUDE

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
        POST to Anthropic /v1/messages and return (response_text, input_tokens, output_tokens).

        No retry logic here — Anthropic's API is reliable. If we get a connection
        error or timeout, it's raised immediately so the caller can decide whether
        to surface an error or degrade gracefully.

        Image handling:
          Full resolution bytes are sent. No resizing. Claude runs on Anthropic's
          infrastructure so VRAM is not a concern. More importantly, this provider
          is called for documents the local model struggled with — often degraded,
          rotated, or low-contrast scans where every pixel of resolution helps.
        """
        url = f"{_ANTHROPIC_API_BASE}/messages"
        headers = self._build_headers()
        content_blocks = self._build_content_blocks(prompt, image_bytes, image_mime_type)

        payload = {
            "model": self.model_name,
            "max_tokens": self._settings.claude_max_tokens,
            "system": _SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": content_blocks}
            ],
            # temperature=0 for deterministic extraction.
            # Claude defaults to 1.0 — must be set explicitly.
            "temperature": 0,
        }

        timeout = self._settings.claude_timeout_seconds

        logger.info(
            "Claude fallback call: model=%s, content_blocks=%d, timeout=%ds",
            self.model_name,
            len(content_blocks),
            timeout,
        )

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)

        except httpx.ConnectError as e:
            raise ProviderUnavailableError(
                provider=self.model_name,
                message=(
                    f"Cannot reach Anthropic API at {_ANTHROPIC_API_BASE}. "
                    f"Check internet connectivity. Original error: {e}"
                ),
            ) from e
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                provider=self.model_name,
                message=(
                    f"Anthropic API did not respond within {timeout}s. "
                    f"Consider increasing CLAUDE_TIMEOUT_SECONDS for complex documents. "
                    f"Original error: {e}"
                ),
            ) from e

        # Handle HTTP-level errors with actionable messages
        if response.status_code == 401:
            raise ProviderResponseError(
                provider=self.model_name,
                status_code=401,
                message=(
                    "Invalid API key. Check ANTHROPIC_API_KEY in your .env file. "
                    "Get a key at https://console.anthropic.com/settings/keys"
                ),
            )
        if response.status_code == 429:
            raise ProviderResponseError(
                provider=self.model_name,
                status_code=429,
                message=(
                    "Anthropic rate limit hit. The fallback is being called too frequently. "
                    "Consider raising CONFIDENCE_THRESHOLD to reduce Claude call volume."
                ),
            )
        if response.status_code == 529:
            raise ProviderResponseError(
                provider=self.model_name,
                status_code=529,
                message="Anthropic API is overloaded. Retry later.",
            )
        if response.status_code != 200:
            raise ProviderResponseError(
                provider=self.model_name,
                status_code=response.status_code,
                message=response.text[:500],
            )

        data = response.json()

        # Extract text from the content array
        response_text = self._extract_text_from_response(data)
        if not response_text:
            raise ProviderResponseError(
                provider=self.model_name,
                status_code=200,
                message=(
                    f"Claude returned no text content. "
                    f"Stop reason: {data.get('stop_reason')}. "
                    f"Content blocks: {data.get('content')}"
                ),
            )

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")

        logger.info(
            "Claude response received. input_tokens=%s, output_tokens=%s, stop_reason=%s",
            input_tokens,
            output_tokens,
            data.get("stop_reason"),
        )

        # Warn if the response was cut off — max_tokens too low for this document
        if data.get("stop_reason") == "max_tokens":
            logger.warning(
                "Claude response hit max_tokens limit (%d). "
                "Some fields may be missing from the response. "
                "Consider increasing CLAUDE_MAX_TOKENS.",
                self._settings.claude_max_tokens,
            )

        return response_text, input_tokens, output_tokens

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Return True if the Anthropic API is reachable and the configured model is available.

        Uses GET /v1/models — a free, read-only endpoint that:
          1. Validates the API key (401 if invalid)
          2. Confirms the model name exists in Anthropic's model catalogue
          3. Is fast (no token generation)

        Returns False on any failure rather than raising, so the router can
        decide to proceed without Claude (local-only mode) rather than crashing.
        """
        url = f"{_ANTHROPIC_API_BASE}/models"

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(url, headers=self._build_headers())

            if response.status_code == 401:
                logger.error(
                    "Claude health check failed: invalid API key. "
                    "Check ANTHROPIC_API_KEY in your .env file."
                )
                return False

            if response.status_code != 200:
                logger.error(
                    "Claude health check failed: HTTP %d from %s",
                    response.status_code,
                    url,
                )
                return False

            data = response.json()

            # /v1/models returns {"data": [{"id": "claude-...", ...}, ...]}
            available_model_ids = [m.get("id", "") for m in data.get("data", [])]

            if self.model_name not in available_model_ids:
                logger.warning(
                    "Claude health check: model '%s' not found in available models. "
                    "Available: %s. "
                    "Check CLAUDE_MODEL in your .env file.",
                    self.model_name,
                    available_model_ids,
                )
                return False

            logger.debug(
                "Claude health check passed. Model '%s' is available.", self.model_name
            )
            return True

        except httpx.ConnectError:
            logger.error(
                "Claude health check failed: cannot reach %s. "
                "Check internet connectivity.", _ANTHROPIC_API_BASE
            )
            return False
        except httpx.TimeoutException:
            logger.error("Claude health check timed out after 8s.")
            return False
        except Exception as e:
            logger.error("Claude health check raised unexpected error: %s", e)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict:
        """
        Construct the required Anthropic API headers.

        x-api-key:        Authentication.
        anthropic-version: Required. Locks the API contract to a specific version.
                          Pinned to 2023-06-01 — the stable release. Update
                          intentionally when you want new API features.
        content-type:     Standard JSON.
        """
        return {
            "x-api-key": self._settings.anthropic_api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    @staticmethod
    def _build_content_blocks(
        prompt: str,
        image_bytes: Optional[bytes],
        image_mime_type: Optional[str],
    ) -> list:
        """
        Build the Anthropic Messages API content array.

        Anthropic's vision API expects a typed content block array, not a flat
        images list (that's Ollama's format). Each block has a 'type' field:
          {"type": "image", "source": {"type": "base64", ...}}
          {"type": "text", "text": "..."}

        Block ordering: image first, then text.
        Anthropic's documentation recommends this order for vision tasks —
        the model processes the image before reading the extraction instructions,
        which improves spatial reasoning over the document layout.

        Image bytes are sent at full resolution (no resizing).
        Claude handles large images server-side. Sending full resolution
        is deliberate — this provider is called for difficult documents
        where visual detail may be the difference between correct and wrong.
        """
        import base64

        blocks = []

        if image_bytes is not None:
            if not image_mime_type:
                logger.warning(
                    "image_bytes provided to Claude without image_mime_type. "
                    "Defaulting to image/jpeg — set image_mime_type for accuracy."
                )
                image_mime_type = "image/jpeg"

            encoded = base64.b64encode(image_bytes).decode("utf-8")
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_mime_type,
                    "data": encoded,
                },
            })

            logger.debug(
                "Image block added: mime_type=%s, encoded_size=%d chars",
                image_mime_type,
                len(encoded),
            )

        blocks.append({
            "type": "text",
            "text": prompt,
        })

        return blocks

    @staticmethod
    def _extract_text_from_response(data: dict) -> str:
        """
        Pull the text content out of an Anthropic Messages API response.

        The response 'content' field is an array of typed blocks:
          [{"type": "text", "text": "..."}, ...]

        We concatenate all text blocks. In practice there's always exactly one,
        but defensive concatenation handles edge cases (tool use interleaved, etc.).
        """
        content_blocks = data.get("content", [])
        text_parts = [
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        ]
        return "".join(text_parts).strip()