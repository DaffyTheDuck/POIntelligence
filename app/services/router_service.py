"""
app/services/router_service.py

The decision engine. All nine architecture decisions converge here.

Flow for every document:
  1. Estimate document quality from surya OCR confidence scores
  2. Check provider health (cached — health checks are slow)
  3. Select primary provider (auto-routed, architecture decision #2)
  4. Run primary extraction
  5. Find fields below confidence threshold (architecture decision #3)
  6. If low-confidence fields exist AND Claude is available:
       → Re-extract only those fields via Claude (target_fields)
       → Detect value disagreement between providers (architecture decision #4)
       → Merge: Claude's extractions replace Ollama's for fallback fields
  7. Return RouterDecision with merged output + routing metadata

What the router does NOT do:
  - It does not build ExtractionResult (that's extraction_service's job)
  - It does not attach bounding boxes (that's ocr_service's job)
  - It does not validate business rules (that's validation_service's job)
  - It does not know about jobs or files — it only knows about providers and fields
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.config import get_settings
from app.models.po_models import (
    FieldExtraction,
    ModelSource,
    ReviewReason,
    CONFIDENCE_THRESHOLD,
    DISAGREEMENT_THRESHOLD,
)
from app.providers.base import (
    BaseProvider,
    ProviderInput,
    ProviderOutput,
    ProviderError,
    ProviderUnavailableError,
    ALL_EXTRACTABLE_FIELDS,
)
from app.providers.ollama_provider import OllamaProvider
from app.providers.claude_provider import ClaudeProvider
from app.services.ocr_service import OCRDocument

logger = logging.getLogger(__name__)

# Document quality score below which we skip Ollama and go straight to Claude.
# Rationale: if surya is averaging <0.6 confidence, the scan is too degraded
# for phi3.5-vision to handle reliably. Don't waste time on a doomed inference.
_QUALITY_THRESHOLD_FOR_LOCAL = 0.60

# Health check cache TTL — 30s balances freshness vs latency.
# A crashed Ollama will be detected within 30s. An HTTP health check takes ~300ms.
_HEALTH_CACHE_TTL_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Router output type
# ---------------------------------------------------------------------------


@dataclass
class RouterDecision:
    """
    The complete output of one routing cycle.

    Consumed by extraction_service to build an ExtractionResult.
    Contains both the merged field data and the metadata needed to populate
    ExtractionResult's routing-provenance fields.
    """
    merged_output: ProviderOutput
    primary_source: ModelSource       # Which model ran first
    fallback_triggered: bool          # True if Claude was called
    fallback_fields: List[str]        # Fields that were re-extracted by Claude
    disagreement_fields: List[str]    # Fields where models produced different values
    document_quality_score: float     # 0-1, from surya average confidence
    ollama_available: bool            # Recorded for audit / monitoring
    claude_available: bool


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class RouterService:
    """
    Orchestrates provider selection, fallback, disagreement detection, and merging.

    Providers are injected via constructor — defaults create real implementations
    but tests can pass mock providers without patching globals.

    Usage:
        router = RouterService()
        decision = await router.route(ocr_doc)
        # decision.merged_output.field_extractions is ready for extraction_service
    """

    def __init__(
        self,
        ollama_provider: Optional[BaseProvider] = None,
        claude_provider: Optional[BaseProvider] = None,
    ) -> None:
        self._ollama: BaseProvider = ollama_provider or OllamaProvider()
        self._claude: BaseProvider = claude_provider or ClaudeProvider()
        self._settings = get_settings()

        # Health check cache: provider_name → (is_healthy, checked_at_unix_timestamp)
        self._health_cache: Dict[str, Tuple[bool, float]] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def route(self, ocr_doc: OCRDocument) -> RouterDecision:
        """
        Run the full routing cycle for one document.

        This is the only method extraction_service calls. Everything else is
        implementation detail hidden inside this class.
        """
        settings = self._settings

        # Step 1: Assess document quality
        quality_score = self._estimate_document_quality(ocr_doc)
        logger.info(
            "Document %s quality score: %.2f (threshold for local=%.2f)",
            ocr_doc.document_id, quality_score, _QUALITY_THRESHOLD_FOR_LOCAL,
        )

        # Step 2: Check provider health (cached)
        ollama_ok, claude_ok = await self._check_provider_health()
        logger.info(
            "Provider health — ollama=%s, claude=%s", ollama_ok, claude_ok
        )

        if not ollama_ok and not claude_ok:
            raise RuntimeError(
                "Both providers are unavailable. "
                "Check that Ollama is running on the Linux machine and "
                "that ANTHROPIC_API_KEY is valid."
            )

        # Step 3: Select primary provider (architecture decision #2 — auto-routed)
        primary = self._select_primary_provider(
            quality_score=quality_score,
            ollama_ok=ollama_ok,
            claude_ok=claude_ok,
        )
        logger.info(
            "Primary provider selected: %s (quality=%.2f, ollama_ok=%s)",
            primary.model_name, quality_score, ollama_ok,
        )

        # Step 4: Build input and run primary extraction
        primary_input = self._build_provider_input(ocr_doc, target_fields=None)

        try:
            primary_output = await primary.extract(primary_input)
        except ProviderError as e:
            logger.error("Primary provider '%s' failed: %s", primary.model_name, e)
            # If primary was Ollama and Claude is available, fall through to Claude
            if primary.source == ModelSource.LOCAL and claude_ok:
                logger.warning("Falling back to Claude as primary due to Ollama failure.")
                primary = self._claude
                primary_output = await primary.extract(primary_input)
            else:
                raise

        logger.info(
            "Primary extraction complete. %d fields extracted, latency=%dms",
            len(primary_output.field_extractions),
            primary_output.latency_ms,
        )

        # Step 5: Identify low-confidence fields (architecture decision #3)
        low_confidence_fields = self._fields_needing_fallback(primary_output)

        if not low_confidence_fields:
            logger.info(
                "All fields above confidence threshold (%.2f). No fallback needed.",
                settings.confidence_threshold,
            )
            return RouterDecision(
                merged_output=primary_output,
                primary_source=primary.source,
                fallback_triggered=False,
                fallback_fields=[],
                disagreement_fields=[],
                document_quality_score=quality_score,
                ollama_available=ollama_ok,
                claude_available=claude_ok,
            )

        # Step 6: Run fallback on low-confidence fields
        logger.info(
            "%d fields below confidence threshold %.2f → escalating to Claude: %s",
            len(low_confidence_fields),
            settings.confidence_threshold,
            low_confidence_fields,
        )

        if not claude_ok:
            logger.warning(
                "Low-confidence fields detected but Claude is unavailable. "
                "Flagging %d fields for human review.",
                len(low_confidence_fields),
            )
            # Flag the low-confidence fields for review since we can't escalate
            flagged_output = self._flag_fields_for_review(
                primary_output, low_confidence_fields, ReviewReason.LOW_CONFIDENCE
            )
            return RouterDecision(
                merged_output=flagged_output,
                primary_source=primary.source,
                fallback_triggered=False,
                fallback_fields=[],
                disagreement_fields=low_confidence_fields,
                document_quality_score=quality_score,
                ollama_available=ollama_ok,
                claude_available=False,
            )

        # Claude is available — run targeted fallback
        fallback_input = self._build_provider_input(
            ocr_doc, target_fields=low_confidence_fields
        )

        try:
            fallback_output = await self._claude.extract(fallback_input)
        except ProviderError as e:
            logger.error("Claude fallback failed: %s. Flagging fields for human review.", e)
            flagged_output = self._flag_fields_for_review(
                primary_output, low_confidence_fields, ReviewReason.LOW_CONFIDENCE
            )
            return RouterDecision(
                merged_output=flagged_output,
                primary_source=primary.source,
                fallback_triggered=False,
                fallback_fields=[],
                disagreement_fields=low_confidence_fields,
                document_quality_score=quality_score,
                ollama_available=ollama_ok,
                claude_available=claude_ok,
            )

        logger.info(
            "Claude fallback complete. %d fields re-extracted, latency=%dms",
            len(fallback_output.field_extractions),
            fallback_output.latency_ms,
        )

        # Step 7: Detect disagreement (architecture decision #4)
        disagreement_fields = self._detect_disagreement(primary_output, fallback_output)
        if disagreement_fields:
            logger.info(
                "Model disagreement detected on %d fields: %s",
                len(disagreement_fields), disagreement_fields,
            )

        # Step 8: Merge primary and fallback outputs
        merged_output = self._merge_outputs(
            primary_output=primary_output,
            fallback_output=fallback_output,
            disagreement_fields=disagreement_fields,
        )

        return RouterDecision(
            merged_output=merged_output,
            primary_source=primary.source,
            fallback_triggered=True,
            fallback_fields=low_confidence_fields,
            disagreement_fields=disagreement_fields,
            document_quality_score=quality_score,
            ollama_available=ollama_ok,
            claude_available=claude_ok,
        )

    # ------------------------------------------------------------------
    # Provider selection — architecture decision #2
    # ------------------------------------------------------------------

    def _select_primary_provider(
        self,
        quality_score: float,
        ollama_ok: bool,
        claude_ok: bool,
    ) -> BaseProvider:
        """
        Select which provider runs first. This is auto-routing — no user input.

        Decision tree:
          Ollama unavailable        → Claude (no choice)
          Document quality < 0.60   → Claude directly (degrade scan, local will fail)
          Otherwise                 → Ollama (local first, cheaper, faster)

        Architecture decision #2: the system decides routing, not the user.
        The quality threshold is the key mechanism — if surya is averaging
        low confidence, we don't waste GPU time on a doomed Ollama inference.
        """
        if not ollama_ok:
            logger.debug("Selecting Claude as primary: Ollama unavailable.")
            return self._claude

        if quality_score < _QUALITY_THRESHOLD_FOR_LOCAL:
            logger.debug(
                "Selecting Claude as primary: document quality %.2f < threshold %.2f.",
                quality_score, _QUALITY_THRESHOLD_FOR_LOCAL,
            )
            return self._claude

        return self._ollama

    # ------------------------------------------------------------------
    # Confidence analysis — architecture decision #3
    # ------------------------------------------------------------------

    def _fields_needing_fallback(self, output: ProviderOutput) -> List[str]:
        """
        Return field paths where confidence is below the configured threshold.

        These are the fields Claude will re-extract. By passing them as
        target_fields, the fallback prompt only asks for these fields —
        not the entire document. This is the key cost-reduction mechanism.

        Fields with None value are included — they weren't found at all.
        """
        threshold = self._settings.confidence_threshold
        return [
            field_path
            for field_path, extraction in output.field_extractions.items()
            if extraction.confidence < threshold
        ]

    # ------------------------------------------------------------------
    # Disagreement detection — architecture decision #4
    # ------------------------------------------------------------------

    def _detect_disagreement(
        self,
        primary_output: ProviderOutput,
        fallback_output: ProviderOutput,
    ) -> List[str]:
        """
        Find fields where Ollama and Claude extracted different values.

        Disagreement is declared when ALL of the following are true:
          1. Both providers extracted a non-None value for the field
          2. The values differ after normalisation (case/whitespace insensitive)
          3. The confidence delta exceeds DISAGREEMENT_THRESHOLD (0.15)

        Condition 3 matters: if both models return different values but are both
        high-confidence, that's a strong signal something is genuinely ambiguous.
        If one model is low-confidence, the difference is expected — Claude's
        value wins without being flagged as a disagreement.

        Architecture decision #4: flag for human review, never guess.
        """
        disagreement_fields: List[str] = []

        for field_path, fallback_ext in fallback_output.field_extractions.items():
            primary_ext = primary_output.field_extractions.get(field_path)

            if primary_ext is None:
                continue  # Field wasn't in primary output — no comparison possible

            # Both must have non-None values to count as a disagreement
            if primary_ext.value is None or fallback_ext.value is None:
                continue

            # Normalise for comparison — ignore case and whitespace differences
            primary_norm = _normalise_value(primary_ext.value)
            fallback_norm = _normalise_value(fallback_ext.value)

            if primary_norm == fallback_norm:
                continue  # Same value — no disagreement

            # Values differ — check confidence delta
            conf_delta = abs(primary_ext.confidence - fallback_ext.confidence)
            if conf_delta > DISAGREEMENT_THRESHOLD:
                logger.debug(
                    "Disagreement on '%s': primary=%r (%.2f) vs claude=%r (%.2f), delta=%.2f",
                    field_path,
                    primary_ext.value, primary_ext.confidence,
                    fallback_ext.value, fallback_ext.confidence,
                    conf_delta,
                )
                disagreement_fields.append(field_path)

        return disagreement_fields

    # ------------------------------------------------------------------
    # Output merging
    # ------------------------------------------------------------------

    def _merge_outputs(
        self,
        primary_output: ProviderOutput,
        fallback_output: ProviderOutput,
        disagreement_fields: List[str],
    ) -> ProviderOutput:
        """
        Merge primary (Ollama) and fallback (Claude) field extractions.

        Merge rules per field:
          Not in fallback output  → keep primary as-is
          In fallback, no disagree → use Claude's extraction (it ran because primary was low)
          In fallback, disagree    → use Claude's value but flag MODEL_DISAGREEMENT

        The merged output's source is recorded as LOCAL (primary was Ollama)
        because that's what ran first. The fallback fields carry CLAUDE as their
        source individually inside FieldExtraction.source_model.
        """
        merged_extractions: Dict[str, FieldExtraction] = dict(
            primary_output.field_extractions
        )

        for field_path, claude_ext in fallback_output.field_extractions.items():
            if field_path in disagreement_fields:
                # Models disagreed — keep Claude's value but flag for human review
                merged_extractions[field_path] = claude_ext.model_copy(update={
                    "flagged_for_review": True,
                    "review_reason": ReviewReason.MODEL_DISAGREEMENT,
                })
            else:
                # Claude ran and agreed (or primary had no value) — use Claude's result
                merged_extractions[field_path] = claude_ext

        # Rebuild a ProviderOutput with merged extractions.
        # Latency is the sum of both calls — total wall clock for this document.
        return ProviderOutput(
            field_extractions=merged_extractions,
            model_name=f"{primary_output.model_name}+{fallback_output.model_name}",
            source=primary_output.source,
            raw_response=(
                f"[PRIMARY]\n{primary_output.raw_response}\n\n"
                f"[FALLBACK]\n{fallback_output.raw_response}"
            ),
            prompt_tokens=(primary_output.prompt_tokens or 0) + (fallback_output.prompt_tokens or 0),
            completion_tokens=(primary_output.completion_tokens or 0) + (fallback_output.completion_tokens or 0),
            latency_ms=primary_output.latency_ms + fallback_output.latency_ms,
            parse_errors=primary_output.parse_errors + fallback_output.parse_errors,
        )

    # ------------------------------------------------------------------
    # Health checking — cached
    # ------------------------------------------------------------------

    async def _check_provider_health(self) -> Tuple[bool, bool]:
        """
        Return (ollama_healthy, claude_healthy) with 30s caching.

        Health checks take 300-500ms each. Caching prevents adding 600-1000ms
        to every document request. 30s TTL means a crashed provider is detected
        quickly enough without paying the check cost on every request.
        """
        now = time.monotonic()

        ollama_ok = await self._cached_health_check(self._ollama, now)
        claude_ok = await self._cached_health_check(self._claude, now)

        return ollama_ok, claude_ok

    async def _cached_health_check(
        self, provider: BaseProvider, now: float
    ) -> bool:
        """Return cached health status, or run a fresh check if cache is stale."""
        name = provider.model_name
        cached = self._health_cache.get(name)

        if cached is not None:
            is_healthy, checked_at = cached
            if now - checked_at < _HEALTH_CACHE_TTL_SECONDS:
                return is_healthy

        # Cache miss or stale — run a fresh check
        try:
            is_healthy = await provider.health_check()
        except Exception as e:
            logger.warning("Health check for '%s' raised: %s", name, e)
            is_healthy = False

        self._health_cache[name] = (is_healthy, now)
        return is_healthy

    def invalidate_health_cache(self) -> None:
        """
        Force fresh health checks on the next route() call.
        Call this after manually restarting Ollama or rotating the API key.
        """
        self._health_cache.clear()
        logger.debug("Health check cache cleared.")

    # ------------------------------------------------------------------
    # Document quality estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_document_quality(ocr_doc: OCRDocument) -> float:
        """
        Estimate document quality as the average surya OCR confidence across all pages.

        Returns a float in [0, 1]:
          ~0.9+ → clean digital PDF, all text lines high confidence
          ~0.7  → decent scan, some noise
          ~0.5  → degraded scan, lots of uncertain regions
          ~0.0  → surya found no text (blank page, image-only with no text)

        No text lines → score 0.5 (unknown, not definitely bad).
        Used by _select_primary_provider to decide whether Ollama can handle it.
        """
        all_lines = ocr_doc.all_text_lines
        if not all_lines:
            # surya found nothing — document might be image-only or blank.
            # Return 0.5 (neutral) rather than 0.0 (catastrophic).
            logger.debug("No text lines detected by surya — quality unknown, score=0.5")
            return 0.5

        avg_confidence = sum(line.ocr_confidence for line in all_lines) / len(all_lines)
        logger.debug(
            "Quality estimate: %d text lines, avg_confidence=%.3f",
            len(all_lines), avg_confidence
        )
        return round(avg_confidence, 3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_provider_input(
        ocr_doc: OCRDocument,
        target_fields: Optional[List[str]],
    ) -> ProviderInput:
        """Construct a ProviderInput from an OCRDocument."""
        image_bytes, image_mime_type = ocr_doc.primary_image

        return ProviderInput(
            document_id=ocr_doc.document_id,
            ocr_text=ocr_doc.full_text,
            image_bytes=image_bytes or None,
            image_mime_type=image_mime_type if image_bytes else None,
            page_count=len(ocr_doc.pages),
            target_fields=target_fields,
        )

    @staticmethod
    def _flag_fields_for_review(
        output: ProviderOutput,
        fields_to_flag: List[str],
        reason: ReviewReason,
    ) -> ProviderOutput:
        """
        Return a copy of output with specified fields marked for human review.
        Used when Claude fallback is unavailable and we can't escalate.
        """
        updated = dict(output.field_extractions)
        for field_path in fields_to_flag:
            if field_path in updated:
                updated[field_path] = updated[field_path].model_copy(update={
                    "flagged_for_review": True,
                    "review_reason": reason,
                })

        return output.model_copy(update={"field_extractions": updated})


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _normalise_value(value) -> str:
    """
    Normalise an extracted value for equality comparison in disagreement detection.

    Handles the common case where both models extract the same semantic value
    but with different formatting:
      "ACME Corporation" vs "Acme Corporation" → same
      "2024-01-15" vs "2024-01-15" → same
      "1,234.56" vs "1234.56" → same (comma stripping)
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    # Remove commas (number formatting), extra whitespace, trailing punctuation
    text = text.replace(",", "").replace(".", "").strip()
    text = " ".join(text.split())
    return text