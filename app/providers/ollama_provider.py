"""
app/providers/ollama_provider.py

Concrete provider for phi3.5-vision running locally via Ollama on the Linux GPU machine.

Implements the three abstract members from BaseProvider:
  - model_name  — "phi3.5-vision" (or whatever OLLAMA_MODEL is set to)
  - source      — ModelSource.LOCAL
  - extract_raw — POST to Ollama /api/chat, return raw response text

Everything else (prompt building, JSON parsing, confidence heuristics) is
inherited from BaseProvider and shared with ClaudeProvider.

Hardware context:
  GTX 1650, 4GB VRAM. phi3.5-vision at 4-bit quantization occupies ~2.1GB.
  With KV cache for a 4096-token context: up to ~3.5GB peak.
  Leaving ~500MB headroom — enough, but only if we don't feed huge images.
  Image resizing (max 1024px) is mandatory, not optional.
"""

from __future__ import annotations

import asyncio
import base64
import io
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

# Maximum image dimension before resizing.
# 1024px is the sweet spot for phi3.5-vision — enough detail to read text,
# small enough to keep VRAM pressure low.
_MAX_IMAGE_DIMENSION = 1024

# Ollama generation parameters — applied to every request.
# num_ctx: context window in tokens. 4096 handles most PO documents.
# num_predict: max output tokens. 2048 is generous for a JSON extraction response.
# temperature: 0 = fully deterministic. Never deviate from this for extraction.
_OLLAMA_OPTIONS = {
    "temperature": 0,
    "num_ctx": 8192,
    "num_predict": 4096,   # 32 fields × ~50 tokens each + line_items easily exceeds 2048
    "top_p": 1.0,
    "repeat_penalty": 1.0,
    "cache_prompt": False,  # Force fresh evaluation — cached responses return stale
                            # extractions from previous documents,
    "think": False
}


class OllamaProvider(BaseProvider):
    """
    Extracts PO fields using phi3.5-vision via a local Ollama server.

    Intended to run first on every document (architecture decision #2).
    If confidence falls below threshold, ClaudeProvider takes over the
    low-scoring fields (architecture decision #3).

    Thread safety: httpx.AsyncClient is created fresh per-request.
    This avoids connection pool contention when Celery runs multiple workers,
    and keeps VRAM usage predictable (one inference at a time per Celery config).
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._settings.ollama_model

    @property
    def source(self) -> ModelSource:
        return ModelSource.LOCAL

    @property
    def is_vision_model(self) -> bool:
        """
        Return True if the configured Ollama model supports image input.

        Text-only models (qwen2.5, mistral, llama3, etc.) must NOT receive
        image bytes — Ollama will either error or silently ignore them.
        Vision models (llava, phi3.5-vision, moondream, etc.) expect images.

        Note: llava-phi3 was evaluated but disabled (hardcoded False) because
        its 4096-token context window was consumed entirely by image patch tokens,
        leaving no room for the 32-field prompt + JSON response — producing
        garbled output at 354s inference time.

        moondream uses a much more efficient vision encoder (SigLIP), fits in
        ~1.8GB VRAM, and natively supports structured JSON output — making it
        the correct choice for local vision extraction on 4GB hardware.

        Checked by _build_chat_message to conditionally attach image data.
        """
        vision_keywords = ["llava", "vision", "moondream", "bakllava"]
        return any(kw in self.model_name.lower() for kw in vision_keywords)

    # ------------------------------------------------------------------
    # Core API call — the only method subclasses need to implement
    # ------------------------------------------------------------------

    async def extract_raw(
        self,
        prompt: str,
        image_bytes: Optional[bytes],
        image_mime_type: str | None,
    ) -> tuple[str, int | None, int | None]:
        """
        POST to Ollama /api/chat and return (response_text, prompt_tokens, completion_tokens).

        Uses the chat endpoint rather than /api/generate because Ollama applies
        the model's native chat template automatically — important for phi3.5-vision
        which uses a specific <|user|> / <|assistant|> prompt format internally.

        Retry logic:
          - Retries on ConnectError and TimeoutException (transient — server busy)
          - Does NOT retry on HTTP error responses (4xx/5xx) — those are input problems
          - Exponential backoff: 1s, 2s, 4s, ...
        """
        url = f"{self._settings.ollama_base_url}/api/chat"
        timeout = self._settings.ollama_timeout_seconds
        max_retries = self._settings.ollama_max_retries

        # Build the message content
        message = self._build_chat_message(prompt, image_bytes, image_mime_type)
        payload = {
            "model": self.model_name,
            "messages": [message],
            "stream": False,
            "options": _OLLAMA_OPTIONS,
            "keep_alive": 0,  # Unload model immediately after inference so surya
                              # has full VRAM available on the next request.
        }

        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                wait_seconds = 2 ** (attempt - 1)  # 1s, 2s, 4s
                logger.warning(
                    "Ollama retry %d/%d for document after %ds backoff. Error: %s",
                    attempt,
                    max_retries,
                    wait_seconds,
                    last_error,
                )
                await asyncio.sleep(wait_seconds)

            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload)

                if response.status_code != 200:
                    # HTTP error — not worth retrying, raise immediately
                    raise ProviderResponseError(
                        provider=self.model_name,
                        status_code=response.status_code,
                        message=response.text[:500],
                    )

                data = response.json()
                response_text = data.get("message", {}).get("content", "")
                prompt_tokens = data.get("prompt_eval_count")
                completion_tokens = data.get("eval_count")

                if not response_text:
                    raise ProviderResponseError(
                        provider=self.model_name,
                        status_code=200,
                        message="Ollama returned an empty response content",
                    )

                logger.info(
                    "Ollama inference complete. prompt_tokens=%s completion_tokens=%s",
                    prompt_tokens,
                    completion_tokens,
                )
                return response_text, prompt_tokens, completion_tokens

            except httpx.ConnectError as e:
                last_error = ProviderUnavailableError(
                    provider=self.model_name,
                    message=(
                        f"Cannot connect to Ollama at {self._settings.ollama_base_url}. "
                        f"Is it running? Is the LAN IP correct? Original error: {e}"
                    ),
                )
            except httpx.TimeoutException as e:
                last_error = ProviderTimeoutError(
                    provider=self.model_name,
                    message=(
                        f"Ollama did not respond within {timeout}s. "
                        f"Consider increasing OLLAMA_TIMEOUT_SECONDS. Original error: {e}"
                    ),
                )
            except (ProviderResponseError, ProviderUnavailableError, ProviderTimeoutError):
                raise  # Don't swallow typed provider errors
            except Exception as e:
                last_error = ProviderUnavailableError(
                    provider=self.model_name,
                    message=f"Unexpected error during Ollama request: {e}",
                )

        # Exhausted all retries
        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Return True if Ollama is running AND the configured model is available.

        Two failure modes we distinguish:
          1. Server unreachable  → log error, return False (router skips this provider)
          2. Server up, model missing → log warning with pull instruction, return False

        Checking model presence matters: Ollama can be running but phi3.5-vision
        not yet pulled. On first boot of the Linux machine, you need to run:
          ollama pull phi3.5-vision
        before the health check passes.
        """
        url = f"{self._settings.ollama_base_url}/api/tags"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)

            if response.status_code != 200:
                logger.error(
                    "Ollama health check failed: HTTP %d from %s",
                    response.status_code,
                    url,
                )
                return False

            data = response.json()
            available_models = [m.get("name", "") for m in data.get("models", [])]

            # Ollama tags include the variant suffix, e.g. "phi3.5-vision:latest"
            # Match if our model name is a prefix of any available tag
            model_available = any(
                tag.startswith(self.model_name) for tag in available_models
            )

            if not model_available:
                logger.warning(
                    "Ollama is running but model '%s' is not available. "
                    "Available models: %s. "
                    "Run: ollama pull %s",
                    self.model_name,
                    available_models,
                    self.model_name,
                )
                return False

            logger.debug("Ollama health check passed. Model '%s' is available.", self.model_name)
            return True

        except httpx.ConnectError:
            logger.error(
                "Ollama health check failed: cannot connect to %s. "
                "Is Ollama running on the Linux machine? "
                "Is OLLAMA_BASE_URL set to the correct LAN IP?",
                self._settings.ollama_base_url,
            )
            return False
        except httpx.TimeoutException:
            logger.error("Ollama health check timed out after 5s.")
            return False
        except Exception as e:
            logger.error("Ollama health check raised unexpected error: %s", e)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_chat_message(
        self,
        prompt: str,
        image_bytes: Optional[bytes],
        image_mime_type: str | None,
    ) -> dict:
        """
        Build the Ollama chat message dict.

        For text-only requests: {"role": "user", "content": prompt}
        For vision requests:    {"role": "user", "content": prompt, "images": ["base64..."]}

        Images are resized before encoding to protect VRAM (see _resize_image).
        Ollama expects raw base64 (no data: URI prefix).
        """
        message: dict = {"role": "user", "content": prompt}

        if image_bytes is not None and self.is_vision_model:
            resized_bytes = self._resize_image(image_bytes, image_mime_type)
            encoded = base64.b64encode(resized_bytes).decode("utf-8")
            message["images"] = [encoded]
            logger.debug(
                "Image attached: original=%d bytes, resized=%d bytes",
                len(image_bytes),
                len(resized_bytes),
            )

        return message

    @staticmethod
    def _resize_image(
        image_bytes: bytes,
        mime_type: str | None,
    ) -> bytes:
        """
        Resize image so its longest edge is at most _MAX_IMAGE_DIMENSION pixels.

        Why this is non-negotiable on a 4GB VRAM card:
          A 2400×3300 scanned PO at 300 DPI is ~23MB when decoded.
          phi3.5-vision's vision encoder processes this as a sequence of patches —
          at full resolution that's ~2000+ patches, consuming ~800MB of VRAM
          on top of the model weights. On a 4GB card, that's the difference
          between a successful inference and an OOM kill.

          At 1024px longest edge: ~400 patches, ~160MB VRAM for the image.
          Text on a PO is still clearly readable at this resolution.

        Falls back to returning the original bytes if PIL is unavailable or
        if resizing fails — better to try with the large image than to crash.
        """
        try:
            from PIL import Image  # type: ignore[import]

            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size

            if max(width, height) <= _MAX_IMAGE_DIMENSION:
                # Already within bounds — return as-is, avoid recompression loss
                return image_bytes

            # Calculate new size preserving aspect ratio
            scale = _MAX_IMAGE_DIMENSION / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            resized = img.resize((new_width, new_height), Image.LANCZOS)

            # Convert back to bytes
            output_format = _mime_to_pil_format(mime_type) or img.format or "JPEG"
            buffer = io.BytesIO()
            resized.save(buffer, format=output_format, quality=90)
            buffer.seek(0)

            logger.debug(
                "Resized image: (%d×%d) → (%d×%d), format=%s",
                width, height, new_width, new_height, output_format,
            )
            return buffer.read()

        except ImportError:
            logger.warning(
                "PIL not installed — cannot resize image before Ollama inference. "
                "Install with: pip install Pillow. "
                "Proceeding with original size (VRAM risk on large documents)."
            )
            return image_bytes
        except Exception as e:
            logger.warning(
                "Image resize failed (%s) — proceeding with original bytes.", e
            )
            return image_bytes


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _mime_to_pil_format(mime_type: str | None) -> str | None:
    """Map MIME type string to PIL format identifier."""
    if mime_type is None:
        return None
    _map = {
        "image/jpeg": "JPEG",
        "image/jpg": "JPEG",
        "image/png": "PNG",
        "image/tiff": "TIFF",
        "image/webp": "WEBP",
        "application/pdf": None,  # PDFs converted upstream before reaching here
    }
    return _map.get(mime_type.lower())