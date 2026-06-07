"""
app/services/extraction_service.py

Pipeline orchestrator. The only service that knows about all other services.

Sequence for every document:
  1. ocr_service.process_document()    → OCRDocument (text + bounding boxes)
  2. router_service.route()            → RouterDecision (merged field extractions)
  3. ocr_service.attach_bounding_boxes() → field_extractions with pixel locations
  4. _build_po_data()                  → nested POData from flat field dict
  5. validation_service.validate()     → ValidationFlag list
  6. _assemble_result()                → final ExtractionResult

The mapping in step 4 (flat dot-paths → nested Pydantic models) is the most
failure-prone step. Every defensive measure is there because LLMs produce
unexpected formats: numbers as strings, dates in wrong formats, line_items
as a JSON string instead of a parsed array, etc.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app.models.po_models import (
    BuyerInfo,
    ExtractionResult,
    FieldExtraction,
    LineItem,
    ModelSource,
    POData,
    POTotals,
    ReviewReason,
    ValidationFlag,
    VendorInfo,
)
from app.services.ocr_service import OCRDocument, OCRService
from app.services.router_service import RouterDecision, RouterService
from app.services.validation_service import ValidationService

logger = logging.getLogger(__name__)


class ExtractionService:
    """
    Sequences the full extraction pipeline and assembles ExtractionResult.

    All dependencies injected — defaults create real implementations,
    tests can pass mocks without patching globals.
    """

    def __init__(
        self,
        ocr_service: Optional[OCRService] = None,
        router_service: Optional[RouterService] = None,
        validation_service: Optional[ValidationService] = None,
    ) -> None:
        self._ocr = ocr_service or OCRService()
        self._router = router_service or RouterService()
        self._validation = validation_service or ValidationService()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def extract(
        self,
        file_path: str,
        document_id: str,
        mime_type: str,
    ) -> ExtractionResult:
        """
        Run the complete extraction pipeline for one document.

        Called by:
          - document_service (synchronous upload path)
          - Celery task (async email path)

        Both paths call this method identically — the async/sync distinction
        lives in the callers, not here (architecture decision #1).

        Raises:
          RuntimeError if both providers are unavailable.
          Any unhandled exception propagates — callers handle job failure.
        """
        started_at = datetime.now(timezone.utc)
        logger.info("Extraction started for document %s (%s)", document_id, mime_type)

        # ── Stage 1: OCR ────────────────────────────────────────────────
        ocr_doc = await self._ocr.process_document(
            file_path=file_path,
            document_id=document_id,
            mime_type=mime_type,
        )
        logger.info(
            "OCR complete: %d pages, %d text lines, full_text=%d chars",
            len(ocr_doc.pages),
            len(ocr_doc.all_text_lines),
            len(ocr_doc.full_text),
        )

        # ── Stage 2: Route + Extract ────────────────────────────────────
        decision: RouterDecision = await self._router.route(ocr_doc)
        logger.info(
            "Routing complete: primary=%s, fallback_triggered=%s, "
            "fallback_fields=%d, disagreement_fields=%d",
            decision.primary_source.value,
            decision.fallback_triggered,
            len(decision.fallback_fields),
            len(decision.disagreement_fields),
        )

        # ── Stage 3: Attach bounding boxes ──────────────────────────────
        field_extractions = await self._ocr.attach_bounding_boxes(
            decision.merged_output.field_extractions,
            ocr_doc,
        )

        # ── Stage 3b: Attach bboxes for individual line items ────────────
        # The 'line_items' field is an array — no single bbox covers it.
        # Find each item's description in the OCR text and create individual
        # field paths (line_items.0.description, line_items.1.description, ...)
        # so the frontend can highlight each product row in the document.
        field_extractions = await self._attach_line_item_bboxes(
            field_extractions, ocr_doc
        )

        # ── Stage 4: Map flat fields → nested POData ────────────────────
        po_data, mapping_warnings = self._build_po_data(field_extractions)
        for warning in mapping_warnings:
            logger.warning("Mapping warning [%s]: %s", document_id, warning)

        # ── Stage 5: Validate business rules ────────────────────────────
        validation_flags = self._validation.validate(po_data, field_extractions)
        logger.info(
            "Validation complete: %d flags (%d errors, %d warnings)",
            len(validation_flags),
            sum(1 for f in validation_flags if f.severity == "error"),
            sum(1 for f in validation_flags if f.severity == "warning"),
        )

        # ── Stage 6: Assemble ExtractionResult ──────────────────────────
        completed_at = datetime.now(timezone.utc)
        result = self._assemble_result(
            document_id=document_id,
            po_data=po_data,
            field_extractions=field_extractions,
            decision=decision,
            validation_flags=validation_flags,
            started_at=started_at,
            completed_at=completed_at,
        )

        logger.info(
            "Extraction complete: document=%s, overall_confidence=%.2f, "
            "flagged=%d, duration=%dms",
            document_id,
            result.overall_confidence,
            len(result.fields_flagged_for_review),
            result.processing_duration_ms,
        )

        return result

    async def _attach_line_item_bboxes(
        self,
        field_extractions: Dict[str, FieldExtraction],
        ocr_doc: OCRDocument,
    ) -> Dict[str, FieldExtraction]:
        """
        Find bboxes for individual line item descriptions.

        Creates field paths like 'line_items.0.description', 'line_items.1.description'
        so DocumentViewer can highlight each product row in the document.

        Only matches descriptions (the most visually distinctive per-row value).
        Quantities and prices are typically too short/ambiguous to match reliably.
        """
        from app.models.po_models import ModelSource

        all_lines = ocr_doc.all_text_lines
        if not all_lines:
            return field_extractions

        # Get source model from the line_items field if available
        source = (
            field_extractions.get('line_items', FieldExtraction(
                field_name='line_items', value=None,
                confidence=0.9, source_model=ModelSource.GROQ,
            )).source_model
        )

        # Parse line items from the already-extracted field
        line_items_ext = field_extractions.get('line_items')
        if not line_items_ext or not isinstance(line_items_ext.value, list):
            return field_extractions

        for i, item in enumerate(line_items_ext.value):
            description = None
            if isinstance(item, dict):
                description = item.get('description')
            elif hasattr(item, 'description'):
                description = item.description

            if not description or len(str(description).strip()) < 3:
                continue

            desc_str = str(description).strip()
            field_path = f'line_items.{i}.description'

            # Try bbox matching — same three layers as main fields
            bbox, method = self._ocr._resolve_bounding_box(desc_str, all_lines)

            if bbox is None:
                # Layer 3 — LLM spatial hint
                bbox, method = await self._ocr._llm_spatial_hint(
                    field_path, desc_str, all_lines
                )

            if bbox:
                field_extractions[field_path] = FieldExtraction(
                    field_name=field_path,
                    value=desc_str,
                    confidence=0.92,
                    source_model=source,
                    bounding_box=bbox.model_copy(update={'ocr_match_method': method}),
                    flagged_for_review=False,
                )
                logger.debug(
                    "Line item %d bbox: '%s' → %s",
                    i, desc_str[:30], method.value,
                )

        return field_extractions

    # ------------------------------------------------------------------
    # Stage 4: Flat field dict → nested POData
    # ------------------------------------------------------------------

    def _build_po_data(
        self,
        field_extractions: Dict[str, FieldExtraction],
    ) -> Tuple[POData, List[str]]:
        """
        Map the flat dot-path field dict into a nested POData Pydantic model.

        Returns (po_data, list_of_non_fatal_warnings).

        Dot-path convention:
          "po_number"         → po_data.po_number
          "vendor.name"       → po_data.vendor.name
          "totals.grand_total"→ po_data.totals.grand_total
          "line_items"        → po_data.line_items   (special — array)

        The mapping is defensive throughout. An LLM can return:
          - Numbers as strings ("1234.56" instead of 1234.56)
          - Dates in wrong formats ("Jan 15, 2024" instead of "2024-01-15")
          - line_items as a JSON string instead of a parsed list
          - Unexpected keys that don't exist in the schema

        All of these are logged as warnings and recovered from, not raised.
        """
        warnings: List[str] = []

        # Collect non-None extracted values by field path
        values: Dict[str, Any] = {}
        for field_path, extraction in field_extractions.items():
            if extraction.value is not None:
                values[field_path] = extraction.value

        # ── Top-level scalar fields ──────────────────────────────────────
        top_level_fields = {
            "po_number", "po_date", "due_date", "payment_terms",
            "delivery_terms", "currency", "document_language", "notes",
        }
        top_level: Dict[str, Any] = {}
        for fname in top_level_fields:
            if fname in values:
                top_level[fname] = _coerce_string(values[fname])

        # ── Vendor sub-model ─────────────────────────────────────────────
        vendor_data = self._extract_sub_fields(values, "vendor", [
            "name", "address", "city", "country", "email", "phone", "tax_id", "bank_account"
        ], warnings)
        vendor = VendorInfo(**vendor_data) if vendor_data else None

        # ── Buyer sub-model ──────────────────────────────────────────────
        buyer_data = self._extract_sub_fields(values, "buyer", [
            "name", "address", "department", "contact_person", "email"
        ], warnings)
        buyer = BuyerInfo(**buyer_data) if buyer_data else None

        # ── Totals sub-model ─────────────────────────────────────────────
        totals_data = self._extract_sub_fields(values, "totals", [
            "subtotal", "tax_amount", "shipping", "discount", "grand_total", "currency"
        ], warnings, coerce_numeric=True)
        totals = POTotals(**totals_data) if totals_data else None

        # ── Line items (special case — array) ────────────────────────────
        line_items = self._parse_line_items(
            values.get("line_items"), warnings
        )

        # ── Assemble POData ──────────────────────────────────────────────
        try:
            po_data = POData(
                **top_level,
                vendor=vendor,
                buyer=buyer,
                totals=totals,
                line_items=line_items,
            )
        except Exception as e:
            warnings.append(f"POData construction error: {e}. Using partial data.")
            # Return whatever we have — partial is better than nothing
            po_data = POData(
                vendor=vendor,
                buyer=buyer,
                totals=totals,
                line_items=line_items,
            )

        return po_data, warnings

    @staticmethod
    def _extract_sub_fields(
        values: Dict[str, Any],
        prefix: str,
        field_names: List[str],
        warnings: List[str],
        coerce_numeric: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract sub-model fields from the flat values dict using dot-path convention.

        e.g. prefix="vendor", field_names=["name","email"]
        looks for values["vendor.name"], values["vendor.email"]

        Returns a dict ready to unpack into the Pydantic sub-model constructor.
        """
        result: Dict[str, Any] = {}
        for fname in field_names:
            path = f"{prefix}.{fname}"
            if path not in values:
                continue
            raw = values[path]
            if coerce_numeric and fname not in ("currency",):
                coerced, warning = _coerce_numeric(raw, path)
                if warning:
                    warnings.append(warning)
                result[fname] = coerced
            else:
                result[fname] = _coerce_string(raw)
        return result

    @staticmethod
    def _parse_line_items(
        raw: Any,
        warnings: List[str],
    ) -> List[LineItem]:
        """
        Parse and validate line items from the LLM's extracted value.

        LLMs return line_items in various forms:
          - A proper Python list of dicts (ideal — Pydantic already parsed the JSON)
          - A JSON string "[{...}, {...}]" (common — model put quotes around the array)
          - A single dict instead of a list (model forgot the array wrapper)
          - None or empty (no line items found)

        Each item is individually validated against LineItem schema.
        Bad items are logged and dropped — one malformed item should not
        discard the entire line item table.
        """
        if raw is None:
            return []

        # Handle JSON string case
        if isinstance(raw, str):
            import json
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as e:
                warnings.append(f"line_items: failed to parse JSON string: {e}")
                return []

        # Wrap single dict in a list
        if isinstance(raw, dict):
            raw = [raw]

        if not isinstance(raw, list):
            warnings.append(
                f"line_items: unexpected type {type(raw).__name__}, expected list"
            )
            return []

        items: List[LineItem] = []
        for i, item_data in enumerate(raw):
            if not isinstance(item_data, dict):
                warnings.append(f"line_items[{i}]: not a dict, skipping")
                continue

            # Coerce numeric fields — LLMs often return them as strings
            for numeric_field in ("quantity", "unit_price", "line_total", "tax_rate"):
                if numeric_field in item_data:
                    coerced, warning = _coerce_numeric(
                        item_data[numeric_field], f"line_items[{i}].{numeric_field}"
                    )
                    if warning:
                        warnings.append(warning)
                    item_data[numeric_field] = coerced

            # Drop unknown keys to avoid Pydantic validation errors
            known_fields = {
                "line_number", "description", "part_number", "quantity",
                "unit", "unit_price", "currency", "line_total", "tax_rate", "notes"
            }
            cleaned = {k: v for k, v in item_data.items() if k in known_fields}

            # description is the only required field in LineItem
            if "description" not in cleaned or not cleaned["description"]:
                warnings.append(f"line_items[{i}]: missing 'description', skipping")
                continue

            try:
                items.append(LineItem(**cleaned))
            except Exception as e:
                warnings.append(f"line_items[{i}]: validation error: {e}, skipping")

        logger.debug("Parsed %d/%d line items successfully", len(items), len(raw))
        return items

    # ------------------------------------------------------------------
    # Stage 6: Assemble ExtractionResult
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_result(
        document_id: str,
        po_data: POData,
        field_extractions: Dict[str, FieldExtraction],
        decision: RouterDecision,
        validation_flags: List[ValidationFlag],
        started_at: datetime,
        completed_at: datetime,
    ) -> ExtractionResult:
        """
        Build the final ExtractionResult from all pipeline outputs.

        overall_confidence: minimum confidence across all fields that have a
        non-None value. Weakest-link scoring — one bad field means the whole
        document shouldn't be auto-processed without review. Fields that
        were genuinely not found (None) are excluded from the minimum to
        avoid always scoring 0.0 on documents that omit optional fields.

        fields_flagged_for_review: union of:
          - Fields below confidence threshold (from providers)
          - Fields with model disagreement (from router)
          - Fields with validation errors that require human confirmation
        """
        # Compute overall confidence (weakest non-None field)
        non_null_confidences = [
            ext.confidence
            for ext in field_extractions.values()
            if ext.value is not None
        ]
        overall_confidence = min(non_null_confidences) if non_null_confidences else 0.0

        # Collect all flagged fields (from providers + validation)
        provider_flagged = [
            path for path, ext in field_extractions.items()
            if ext.flagged_for_review
        ]
        validation_flagged = [
            field
            for flag in validation_flags
            if flag.severity == "error"
            for field in flag.affected_fields
        ]
        all_flagged = sorted(set(provider_flagged + validation_flagged))

        # Duration
        duration_ms = int(
            (completed_at - started_at).total_seconds() * 1000
        )

        return ExtractionResult(
            result_id=str(uuid4()),
            document_id=document_id,
            po_data=po_data,
            field_extractions=field_extractions,
            overall_confidence=round(overall_confidence, 3),
            fields_flagged_for_review=all_flagged,
            validation_flags=validation_flags,
            primary_model=decision.primary_source,
            fallback_triggered=decision.fallback_triggered,
            fallback_fields=decision.fallback_fields,
            processing_started_at=started_at,
            processing_completed_at=completed_at,
            processing_duration_ms=duration_ms,
        )


# ---------------------------------------------------------------------------
# Coercion helpers — keep the mapping code clean
# ---------------------------------------------------------------------------


def _coerce_string(value: Any) -> Optional[str]:
    """Convert any value to a stripped string, or None for empty/null."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _coerce_numeric(
    value: Any, field_path: str
) -> Tuple[Optional[float], Optional[str]]:
    """
    Coerce a value to float. Returns (result, warning_or_None).

    Handles common LLM numeric format issues:
      "1,234.56" → 1234.56  (comma thousands separator)
      "USD 1234" → 1234.0   (currency prefix)
      "1234 USD" → 1234.0   (currency suffix)
      "$1,234.56"→ 1234.56  (currency symbol)
      "1.5k"     → None     (non-standard — log warning)
    """
    if value is None:
        return None, None

    if isinstance(value, (int, float)):
        return float(value), None

    if not isinstance(value, str):
        return None, f"{field_path}: unexpected type {type(value).__name__} for numeric field"

    # Strip currency symbols and codes
    import re
    cleaned = re.sub(r"[£$€¥₹]", "", value)          # Currency symbols
    cleaned = re.sub(r"\b[A-Z]{3}\b", "", cleaned)    # 3-letter currency codes
    cleaned = cleaned.replace(",", "").strip()          # Comma thousands separator

    try:
        return float(cleaned), None
    except ValueError:
        return None, (
            f"{field_path}: could not convert {value!r} to float "
            f"(cleaned='{cleaned}'). Field will be None."
        )