"""
app/providers/base.py

Abstract base class defining the contract every LLM provider must satisfy.

The Strategy pattern: router_service holds a BaseProvider reference and calls
.extract() without knowing whether it's talking to Ollama or Claude.
Swap the concrete class → swap the model. Zero other changes required.

Two concrete implementations:
  - OllamaProvider  (app/providers/ollama_provider.py) — local phi3.5-vision
  - ClaudeProvider  (app/providers/claude_provider.py) — Claude API fallback

Shared logic that lives here (not in subclasses):
  - ProviderInput / ProviderOutput data shapes
  - Extraction prompt construction (same instructions for every model)
  - LLM response parsing (strip fences, validate JSON, map to FieldExtraction)
  - Confidence heuristics (boost self-reported score when value appears verbatim in OCR)
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.po_models import (
    BoundingBox,
    FieldExtraction,
    ModelSource,
    ReviewReason,
    CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider I/O types
# ---------------------------------------------------------------------------


class ProviderInput(BaseModel):
    """
    Everything a provider needs to extract fields from one document.

    Design note — why carry both ocr_text AND image_bytes:
      surya gives us clean OCR text, which is the primary input for semantic
      extraction (architecture decision #7). But phi3.5-vision is a vision model —
      it performs better with the image alongside the text. Claude Vision also
      accepts images. Both providers receive both; each uses what it can.

    target_fields controls partial re-extraction:
      None  → extract all PO fields (first pass, or full fallback)
      [...] → extract only these fields (confidence-based fallback — don't
              re-run the whole document just to fix two low-scoring fields)
    """
    document_id: str
    ocr_text: str = Field(..., description="Full OCR-extracted text from surya")
    image_bytes: Optional[bytes] = Field(
        default=None,
        description="Raw image bytes of the document page. None for text-only providers."
    )
    image_mime_type: Optional[str] = Field(
        default=None,
        description="MIME type of image_bytes, e.g. 'image/jpeg'. Required if image_bytes is set."
    )
    page_count: int = Field(default=1, ge=1)
    target_fields: Optional[List[str]] = Field(
        default=None,
        description=(
            "Field paths to extract. None = extract everything. "
            "Set by router_service when re-running only low-confidence fields via fallback. "
            "Format matches ExtractionResult.field_extractions keys: 'vendor.name', "
            "'totals.grand_total', 'line_items.0.unit_price', etc."
        )
    )
    document_language: Optional[str] = Field(
        default=None,
        description="ISO 639-1 hint if language is already known. Helps model accuracy."
    )


class ProviderOutput(BaseModel):
    """
    Raw output from one provider inference pass.

    field_extractions is keyed by field path (same convention as
    ExtractionResult.field_extractions). extraction_service merges this
    into the running result, field by field.
    """
    field_extractions: Dict[str, FieldExtraction] = Field(
        default_factory=dict,
        description="Extracted fields keyed by field path e.g. 'vendor.name'"
    )
    model_name: str
    source: ModelSource
    raw_response: str = Field(
        description="Full raw LLM response string, stored for debugging and audit"
    )
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: int = Field(description="Wall-clock inference time in milliseconds")
    parse_errors: List[str] = Field(
        default_factory=list,
        description="Non-fatal parsing issues — logged but don't fail the request"
    )


# ---------------------------------------------------------------------------
# The fields we extract — single source of truth for the prompt
# ---------------------------------------------------------------------------

# Flat list of all extractable field paths.
# Changing this list changes what every provider asks for — one place to update.
ALL_EXTRACTABLE_FIELDS: List[str] = [
    "po_number",
    "po_date",
    "due_date",
    "payment_terms",
    "delivery_terms",
    "currency",
    "document_language",
    # Vendor
    "vendor.name",
    "vendor.address",
    "vendor.city",
    "vendor.country",
    "vendor.email",
    "vendor.phone",
    "vendor.tax_id",
    # Buyer
    "buyer.name",
    "buyer.address",
    "buyer.department",
    "buyer.contact_person",
    "buyer.email",
    # Totals
    "totals.subtotal",
    "totals.tax_amount",
    "totals.shipping",
    "totals.discount",
    "totals.grand_total",
    "totals.currency",
    # Line items — extracted as a block, parsed separately
    "line_items",
]

# Human-readable descriptions injected into the prompt.
# Reduces hallucination on ambiguous field names.
FIELD_DESCRIPTIONS: Dict[str, str] = {
    "po_number": "The purchase order number or reference number",
    "po_date": "The date the PO was issued (ISO 8601: YYYY-MM-DD)",
    "due_date": "Payment or delivery due date (ISO 8601: YYYY-MM-DD)",
    "payment_terms": "Payment terms string, e.g. 'Net 30', 'Net 60', 'COD', '10 DAYS NET'",
    "delivery_terms": "Delivery/shipping terms, e.g. 'FOB Destination', 'CIF', 'FREIGHT/CARRIAGE PAID'",
    "currency": "3-letter ISO 4217 currency code, e.g. 'USD', 'EUR', 'GBP'. Look for currency symbols ($, £, €) or codes near amounts.",
    "document_language": "ISO 639-1 language code of the document, e.g. 'en', 'de'",
    "vendor.name": "Supplier / vendor company name",
    "vendor.address": (
        "Vendor FULL postal address — combine ALL address lines into one string. "
        "Include street number, street name, city/town, state/county, postcode/zip. "
        "Example: '1 Main Street, Townsville, DH9 OTB'. Do NOT return just the street line."
    ),
    "vendor.city": "Vendor city or town",
    "vendor.country": "Vendor country",
    "vendor.email": "Vendor contact email address",
    "vendor.phone": "Vendor contact phone number including area code",
    "vendor.tax_id": "Vendor tax ID, VAT number, GST number, or EIN",
    "buyer.name": "Buying company name — check SHIP TO, BILL TO, or buyer sections",
    "buyer.address": (
        "Buyer/Ship-to FULL postal address — combine ALL address lines into one string. "
        "Include street, city/town, state/county, postcode/zip. "
        "Example: '44 Shore St, Macduff, AB4 1TX'. Do NOT return just the street line."
    ),
    "buyer.department": "Buyer department, cost centre, or division",
    "buyer.contact_person": "Buyer contact person name",
    "buyer.email": "Buyer contact email",
    "totals.subtotal": "Sum of all line items before tax and shipping (numeric only, no currency symbol)",
    "totals.tax_amount": "Total tax amount (numeric only, not percentage)",
    "totals.shipping": "Shipping or freight cost (numeric only)",
    "totals.discount": "Total discount applied (numeric only)",
    "totals.grand_total": "Final total amount payable (numeric only, no currency symbol)",
    "totals.currency": "Currency for totals — look for currency symbols ($, £, €) near the grand total",
    "line_items": (
        "Array of ALL line items from the products/items table. Each item: "
        "{description, part_number, quantity, unit, unit_price, currency, line_total, tax_rate}. "
        "Include every row in the table — do not skip any."
    ),
}


# ---------------------------------------------------------------------------
# Abstract base provider
# ---------------------------------------------------------------------------


class BaseProvider(ABC):
    """
    Abstract interface every LLM provider must implement.

    Subclasses implement:
      - extract_raw()   — make the actual API call, return raw text response
      - health_check()  — return True if the provider is reachable and ready
      - model_name      — property returning the model identifier string
      - source          — property returning the ModelSource enum value

    Everything else is handled here:
      - Prompt construction  (_build_prompt)
      - Response parsing     (_parse_response)
      - Confidence heuristics (_apply_ocr_heuristics)
      - Timing wrapper       (extract)
    """

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier string for this model, e.g. 'phi3.5-vision' or 'claude-opus-4-5'."""
        ...

    @property
    @abstractmethod
    def source(self) -> ModelSource:
        """ModelSource enum value for this provider."""
        ...

    @abstractmethod
    async def extract_raw(
        self,
        prompt: str,
        image_bytes: Optional[bytes],
        image_mime_type: Optional[str],
    ) -> tuple[str, Optional[int], Optional[int]]:
        """
        Make the API call and return (raw_text_response, prompt_tokens, completion_tokens).

        The base class handles everything else — this method's only job is the
        HTTP call to the model. Keep it as thin as possible.

        Args:
            prompt:          The fully-constructed extraction prompt.
            image_bytes:     Optional raw image bytes. None for text-only inference.
            image_mime_type: MIME type for image_bytes. Required if image_bytes is set.

        Returns:
            Tuple of (response_text, prompt_token_count, completion_token_count).
            Token counts may be None if the provider doesn't report them.

        Raises:
            ProviderUnavailableError: if the provider cannot be reached.
            ProviderTimeoutError:     if the request exceeds the configured timeout.
            ProviderResponseError:    if the provider returns a non-200 / error response.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Return True if the provider is reachable and the model is loaded.

        Called by router_service before routing a document.
        Should be fast (< 2s) — this is a liveness check, not a benchmark.
        """
        ...

    # ------------------------------------------------------------------
    # Public interface — called by extraction_service and router_service
    # ------------------------------------------------------------------

    async def extract(self, input: ProviderInput) -> ProviderOutput:
        """
        Full extraction pipeline for one provider pass.

        Orchestrates: build prompt → call extract_raw → parse response
        → apply OCR confidence heuristics → return ProviderOutput.

        This method is NOT abstract. Subclasses override extract_raw, not this.
        """
        start_ms = int(time.monotonic() * 1000)

        fields_to_extract = input.target_fields or ALL_EXTRACTABLE_FIELDS
        prompt = self._build_prompt(
            ocr_text=input.ocr_text,
            fields=fields_to_extract,
            language_hint=input.document_language,
        )

        logger.info(
            "Provider %s extracting %d fields for document %s",
            self.model_name,
            len(fields_to_extract),
            input.document_id,
        )

        raw_response, prompt_tokens, completion_tokens = await self.extract_raw(
            prompt=prompt,
            image_bytes=input.image_bytes,
            image_mime_type=input.image_mime_type,
        )

        latency_ms = int(time.monotonic() * 1000) - start_ms

        field_extractions, parse_errors = self._parse_response(
            raw_response=raw_response,
            ocr_text=input.ocr_text,
            expected_fields=fields_to_extract,
        )

        return ProviderOutput(
            field_extractions=field_extractions,
            model_name=self.model_name,
            source=self.source,
            raw_response=raw_response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            parse_errors=parse_errors,
        )

    # ------------------------------------------------------------------
    # Shared prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        ocr_text: str,
        fields: List[str],
        language_hint: Optional[str] = None,
    ) -> str:
        """
        Construct the extraction prompt sent to every provider.

        The prompt is identical for Ollama and Claude — the only difference
        between providers is HOW the prompt is delivered (API call format),
        not WHAT it says. Keeping it here means one place to tune extraction
        behaviour.

        Output format: JSON object where each key is a field path and each
        value is an object with 'value' and 'confidence' keys. This structure
        is directly parseable by _parse_response without model-specific logic.
        """
        field_lines = "\n".join(
            f'  "{f}": {{"value": <extracted or null>, "confidence": <0.0-1.0>}}'
            f'  // {FIELD_DESCRIPTIONS.get(f, f)}'
            for f in fields
        )

        language_instruction = (
            f"The document is written in language code '{language_hint}'. "
            if language_hint
            else "Detect the document language automatically. "
        )

        return f"""You are a Purchase Order extraction engine. Extract structured data from the OCR text below.

INSTRUCTIONS:
1. {language_instruction}Extract the exact values as they appear in the document — do not normalise or infer.
2. For each field, provide a confidence score between 0.0 and 1.0:
   - 1.0 = value found verbatim and unambiguously in the text
   - 0.7–0.9 = value found but required minor interpretation (abbreviation, date format, etc.)
   - 0.4–0.6 = value inferred from context, not stated directly
   - 0.0–0.3 = not found, or highly uncertain — prefer null over guessing
3. Use null (not empty string) when a field is not present in the document.
4. For line_items, return a JSON array. Each item must follow the schema shown.
5. Currency codes must be ISO 4217 (3 letters). Dates must be ISO 8601 (YYYY-MM-DD).
6. Return ONLY valid JSON. No markdown, no backticks, no explanation text.

REQUIRED OUTPUT FORMAT:
{{
{field_lines}
}}

OCR TEXT:
---
{ocr_text}
---

Return only the JSON object. Nothing else."""

    # ------------------------------------------------------------------
    # Shared response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw_response: str,
        ocr_text: str,
        expected_fields: List[str],
    ) -> tuple[Dict[str, FieldExtraction], List[str]]:
        """
        Parse the raw LLM response into a dict of FieldExtraction objects.

        Handles the common failure modes:
          - Response wrapped in markdown code fences (```json ... ```)
          - Partial JSON (truncated response due to max_tokens)
          - Extra explanation text before or after the JSON object
          - Fields present in JSON but missing from expected_fields (ignored)
          - Expected fields missing from JSON (treated as null, low confidence)

        Returns (field_extractions_dict, list_of_non_fatal_parse_errors).
        """
        parse_errors: List[str] = []
        field_extractions: Dict[str, FieldExtraction] = {}

        # Step 1: Extract JSON from the response
        cleaned = self._extract_json_from_response(raw_response)
        if cleaned is None:
            parse_errors.append(
                f"Could not extract JSON from response. "
                f"First 200 chars: {raw_response[:200]!r}"
            )
            # Return empty extractions — caller will handle low-confidence result
            return self._empty_extractions(expected_fields), parse_errors

        # Step 2: Parse JSON
        try:
            parsed: Dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as e:
            parse_errors.append(f"JSON parse error: {e}. Attempting partial recovery.")
            parsed = self._attempt_partial_json_recovery(cleaned, parse_errors)

        if not isinstance(parsed, dict):
            parse_errors.append(f"LLM returned non-dict JSON: {type(parsed).__name__}")
            return self._empty_extractions(expected_fields), parse_errors

        # Step 3: Map each field to a FieldExtraction
        for field_path in expected_fields:
            raw_field = parsed.get(field_path)

            if raw_field is None:
                # Field not in response at all — treat as not found
                field_extractions[field_path] = self._make_field_extraction(
                    field_name=field_path,
                    value=None,
                    confidence=0.0,
                    flagged=True,
                    reason=ReviewReason.MISSING_REQUIRED,
                )
                continue

            if not isinstance(raw_field, dict):
                # Model returned a bare value instead of {value, confidence} — recover
                parse_errors.append(
                    f"Field '{field_path}' expected {{value, confidence}} dict, "
                    f"got {type(raw_field).__name__}. Wrapping with confidence=0.5."
                )
                value = raw_field
                confidence = 0.5
            else:
                value = raw_field.get("value")
                confidence = self._coerce_confidence(
                    raw_field.get("confidence"), field_path, parse_errors
                )

            # Step 4: Apply OCR heuristic — boost confidence when value is in OCR text
            confidence = self._apply_ocr_heuristic(value, confidence, ocr_text)

            # Step 5: Flag if below threshold
            flagged = confidence < CONFIDENCE_THRESHOLD
            reason = ReviewReason.LOW_CONFIDENCE if flagged else None

            field_extractions[field_path] = self._make_field_extraction(
                field_name=field_path,
                value=value,
                confidence=confidence,
                flagged=flagged,
                reason=reason,
            )

        return field_extractions, parse_errors

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_from_response(text: str) -> Optional[str]:
        """
        Pull a JSON object out of a response that may contain surrounding text.

        Handles:
          - Clean JSON (model followed instructions perfectly)
          - ```json ... ``` fences (model added markdown despite being told not to)
          - ``` ... ``` fences without language tag
          - Preamble text before the JSON object ("Here is the extraction: {...}")
        """
        text = text.strip()

        # Remove markdown fences
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        # Find the outermost { ... } block
        start = text.find("{")
        if start == -1:
            return None

        # Walk the string to find the matching closing brace
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None  # Unbalanced braces — truncated response

    @staticmethod
    def _attempt_partial_json_recovery(
        text: str, parse_errors: List[str]
    ) -> Dict[str, Any]:
        """
        Last-resort attempt to recover something useful from malformed JSON.

        Strategy: truncate at the last complete top-level key-value pair and
        close the object. Not perfect, but better than returning nothing.
        """
        # Find the last complete "key": value, pair and close the object
        last_comma = text.rfind(",")
        if last_comma > 0:
            truncated = text[:last_comma] + "\n}"
            try:
                recovered = json.loads(truncated)
                parse_errors.append(
                    f"Partial JSON recovery succeeded — {len(recovered)} fields recovered."
                )
                return recovered
            except json.JSONDecodeError:
                pass

        parse_errors.append("Partial JSON recovery failed. Returning empty dict.")
        return {}

    @staticmethod
    def _coerce_confidence(
        raw: Any, field_path: str, parse_errors: List[str]
    ) -> float:
        """
        Coerce whatever the model put in the 'confidence' slot to a valid float in [0, 1].
        Models sometimes return strings ("0.9"), percentages (90), or None.
        """
        if raw is None:
            return 0.5  # Unknown → middle of the road

        try:
            val = float(raw)
        except (TypeError, ValueError):
            parse_errors.append(
                f"Field '{field_path}': invalid confidence value {raw!r}, defaulting to 0.5"
            )
            return 0.5

        # Some models return 0–100 instead of 0.0–1.0
        if val > 1.0:
            val = val / 100.0

        return max(0.0, min(1.0, val))

    @staticmethod
    def _apply_ocr_heuristic(
        value: Any, model_confidence: float, ocr_text: str
    ) -> float:
        """
        Adjust confidence based on whether the extracted value appears in OCR text.

        Boost when found verbatim — the model is almost certainly right.
        No penalty when not found — the model is using vision (reading the image
        directly), not just regurgitating OCR text. Vision extraction is valid
        even when the OCR text is sparse or the value spans multiple lines.

        The old cap (0.70) was too aggressive — it caused correctly extracted
        multi-line values (addresses, compound names) to always fall below the
        0.75 review threshold, flooding the review queue with false positives.

        Ceiling of 0.95 — never fully trust automated extraction.
        """
        if value is None or not isinstance(value, (str, int, float)):
            return model_confidence

        value_str = str(value).strip()
        if len(value_str) < 3:
            return model_confidence

        # Check verbatim match and also check individual parts for multi-line values
        ocr_lower = ocr_text.lower()
        value_lower = value_str.lower()

        # Full value found verbatim → strong boost
        if value_lower in ocr_lower:
            return min(0.95, model_confidence + 0.15)

        # For multi-part values (addresses), check if most parts are in OCR text
        parts = [p.strip().lower() for p in value_str.replace(',', ' ').split()
                 if len(p.strip()) >= 3]
        if parts:
            found_parts = sum(1 for p in parts if p in ocr_lower)
            coverage = found_parts / len(parts)
            if coverage >= 0.5:
                # At least half the words found — partial boost
                return min(0.92, model_confidence + 0.08)

        # Value not found in OCR text at all — model used vision directly.
        # Don't penalise. Vision extraction is valid without OCR text match.
        return model_confidence

    def _make_field_extraction(
        self,
        field_name: str,
        value: Any,
        confidence: float,
        flagged: bool,
        reason: Optional[ReviewReason],
    ) -> FieldExtraction:
        """Construct a FieldExtraction with this provider's source set."""
        return FieldExtraction(
            field_name=field_name,
            value=value,
            confidence=confidence,
            source_model=self.source,
            flagged_for_review=flagged,
            review_reason=reason if flagged else None,
            # bounding_box is None here — ocr_service populates it after extraction
        )

    def _empty_extractions(
        self, fields: List[str]
    ) -> Dict[str, FieldExtraction]:
        """
        Return a dict of zero-confidence extractions for every field.
        Used when parsing fails entirely — signals the caller to escalate.
        """
        return {
            f: self._make_field_extraction(
                field_name=f,
                value=None,
                confidence=0.0,
                flagged=True,
                reason=ReviewReason.MISSING_REQUIRED,
            )
            for f in fields
        }


# ---------------------------------------------------------------------------
# Provider-specific exceptions
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base class for all provider errors."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class ProviderUnavailableError(ProviderError):
    """Provider endpoint is not reachable (connection refused, DNS failure, etc.)."""


class ProviderTimeoutError(ProviderError):
    """Provider did not respond within the configured timeout."""


class ProviderResponseError(ProviderError):
    """Provider returned an error response (non-200 HTTP, API error payload, etc.)."""

    def __init__(self, provider: str, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(provider, f"HTTP {status_code}: {message}")