"""
app/services/validation_service.py

Deterministic business rule validation on extracted PO data.
No LLM inference here — pure Python logic applied after extraction is complete.

Called by extraction_service as the final pipeline stage before assembling
ExtractionResult. Returns a list of ValidationFlag objects that surface in
the UI and gate the export/webhook path via ExtractionResult.is_ready_for_export.

Rule categories:
  1. Required fields      — structural minimum for a usable PO
  2. Totals consistency   — line items → subtotal → grand total maths
  3. Line item integrity  — per-item quantity × unit_price = line_total
  4. Date ordering        — due_date must be after po_date
  5. Currency consistency — mixed currencies flagged as suspicious
  6. Format checks        — email addresses, PO number non-empty

Severity convention:
  "error"   — document should not be auto-exported; human review required
  "warning" — suspicious but potentially legitimate; reviewer should check
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Dict, List, Optional

from app.models.po_models import (
    FieldExtraction,
    LineItem,
    POData,
    ValidationFlag,
)

logger = logging.getLogger(__name__)

# Numeric comparison tolerances for financial figures.
# Absolute: differences ≤ 5 cents are always accepted regardless of total size.
# Relative: differences ≤ 1% of the larger value are accepted for large amounts.
# Both apply simultaneously — whichever permits the larger difference wins.
_ABS_TOLERANCE = 0.05   # 5 cents
_REL_TOLERANCE = 0.01   # 1%

# Required fields: field path → human-readable label for error messages
_REQUIRED_FIELDS: Dict[str, str] = {
    "po_number":   "PO Number",
    "vendor.name": "Vendor Name",
}

# Fields that are strongly expected but whose absence is a warning, not an error
_EXPECTED_FIELDS: Dict[str, str] = {
    "po_date":           "PO Date",
    "totals.grand_total": "Grand Total",
}


class ValidationService:
    """
    Validates extracted POData against business rules.

    Stateless — no constructor state needed. All methods are pure functions
    over their inputs. Safe to share a single instance across requests.
    """

    def validate(
        self,
        po_data: POData,
        field_extractions: Dict[str, FieldExtraction],
    ) -> List[ValidationFlag]:
        """
        Run all validation rules and return the full list of flags.

        The ordering here matters for the reviewer UX — errors first,
        then warnings, roughly in the order a human would check them.
        """
        flags: List[ValidationFlag] = []

        flags.extend(self._check_required_fields(po_data, field_extractions))
        flags.extend(self._check_expected_fields(po_data, field_extractions))
        flags.extend(self._check_totals_consistency(po_data))
        flags.extend(self._check_line_item_integrity(po_data))
        flags.extend(self._check_date_ordering(po_data))
        flags.extend(self._check_currency_consistency(po_data))
        flags.extend(self._check_formats(po_data))

        logger.debug(
            "Validation produced %d flags (%d errors, %d warnings)",
            len(flags),
            sum(1 for f in flags if f.severity == "error"),
            sum(1 for f in flags if f.severity == "warning"),
        )
        return flags

    # ------------------------------------------------------------------
    # Rule 1: Required fields
    # ------------------------------------------------------------------

    @staticmethod
    def _check_required_fields(
        po_data: POData,
        field_extractions: Dict[str, FieldExtraction],
    ) -> List[ValidationFlag]:
        """
        Verify that the absolute minimum fields for a usable PO are present.

        Two error messages per missing field:
          - "Not found in document" — extraction produced None (field absent)
          - "Extracted but needs review" — low confidence, flagged by provider

        This distinction helps the reviewer: the first means search the document,
        the second means look at the highlighted field and confirm the value.
        """
        flags: List[ValidationFlag] = []

        field_value_map = {
            "po_number":   po_data.po_number,
            "vendor.name": po_data.vendor.name if po_data.vendor else None,
        }

        for field_path, label in _REQUIRED_FIELDS.items():
            value = field_value_map.get(field_path)
            extraction = field_extractions.get(field_path)

            if value is None or str(value).strip() == "":
                # Distinguish "not found" from "found but flagged"
                if extraction and extraction.flagged_for_review:
                    message = (
                        f"{label} was extracted but flagged for review — "
                        f"confirm the value before processing."
                    )
                else:
                    message = (
                        f"{label} was not found in the document. "
                        f"This field is required for downstream processing."
                    )

                flags.append(ValidationFlag(
                    rule=f"required_field_missing:{field_path}",
                    message=message,
                    severity="error",
                    affected_fields=[field_path],
                ))

        return flags

    @staticmethod
    def _check_expected_fields(
        po_data: POData,
        field_extractions: Dict[str, FieldExtraction],
    ) -> List[ValidationFlag]:
        """
        Warn on strongly-expected but not strictly-required fields.
        Missing grand total or PO date is unusual enough to flag but not
        necessarily an extraction failure — some internal POs omit them.
        """
        flags: List[ValidationFlag] = []

        field_value_map = {
            "po_date":            po_data.po_date,
            "totals.grand_total": po_data.totals.grand_total if po_data.totals else None,
        }

        for field_path, label in _EXPECTED_FIELDS.items():
            value = field_value_map.get(field_path)
            if value is None or str(value).strip() == "":
                flags.append(ValidationFlag(
                    rule=f"expected_field_missing:{field_path}",
                    message=(
                        f"{label} was not found. Verify the document contains this field."
                    ),
                    severity="warning",
                    affected_fields=[field_path],
                ))

        return flags

    # ------------------------------------------------------------------
    # Rule 2: Totals consistency
    # ------------------------------------------------------------------

    @staticmethod
    def _check_totals_consistency(po_data: POData) -> List[ValidationFlag]:
        """
        Verify the financial totals form a consistent chain.

        Two checks:
          A. Sum of line_item.line_total ≈ totals.subtotal
             (only when both line items and subtotal are present)

          B. subtotal + tax + shipping − discount ≈ grand_total
             (only when all relevant fields are present)

        Tolerances: ±5 cents absolute OR ±1% relative (whichever is larger).
        This handles rounding in both small and large POs.
        """
        flags: List[ValidationFlag] = []
        totals = po_data.totals

        # ── Check A: line items → subtotal ───────────────────────────────
        has_line_totals = any(
            item.line_total is not None for item in po_data.line_items
        )
        has_subtotal = totals is not None and totals.subtotal is not None

        if po_data.line_items and has_line_totals and has_subtotal:
            computed_subtotal = sum(
                item.line_total
                for item in po_data.line_items
                if item.line_total is not None
            )
            declared_subtotal = totals.subtotal  # type: ignore[union-attr]

            if not _within_tolerance(computed_subtotal, declared_subtotal):
                flags.append(ValidationFlag(
                    rule="totals.line_items_sum_mismatch",
                    message=(
                        f"Line items sum to {computed_subtotal:.2f} but "
                        f"subtotal is declared as {declared_subtotal:.2f}. "
                        f"Difference: {abs(computed_subtotal - declared_subtotal):.2f}."
                    ),
                    severity="error",
                    affected_fields=["totals.subtotal", "line_items"],
                ))

        # ── Check B: subtotal + adjustments → grand total ────────────────
        if totals is None:
            return flags

        grand_total = totals.grand_total
        subtotal = totals.subtotal

        if grand_total is not None and subtotal is not None:
            tax = totals.tax_amount or 0.0
            shipping = totals.shipping or 0.0
            discount = totals.discount or 0.0

            computed_grand = subtotal + tax + shipping - discount

            if not _within_tolerance(computed_grand, grand_total):
                components = (
                    f"subtotal({subtotal:.2f}) "
                    f"+ tax({tax:.2f}) "
                    f"+ shipping({shipping:.2f}) "
                    f"− discount({discount:.2f}) "
                    f"= {computed_grand:.2f}"
                )
                flags.append(ValidationFlag(
                    rule="totals.grand_total_mismatch",
                    message=(
                        f"Computed grand total does not match declared grand total. "
                        f"{components}, but grand_total is {grand_total:.2f}. "
                        f"Difference: {abs(computed_grand - grand_total):.2f}."
                    ),
                    severity="error",
                    affected_fields=[
                        "totals.grand_total", "totals.subtotal",
                        "totals.tax_amount", "totals.shipping", "totals.discount",
                    ],
                ))

        return flags

    # ------------------------------------------------------------------
    # Rule 3: Per-line item integrity
    # ------------------------------------------------------------------

    @staticmethod
    def _check_line_item_integrity(po_data: POData) -> List[ValidationFlag]:
        """
        For each line item with quantity + unit_price + line_total:
        verify quantity × unit_price ≈ line_total.

        Severity: warning, not error. Some POs include bulk discounts or
        minimum order adjustments that legitimately break the simple multiplication.
        Flagging as warning surfaces it for review without blocking export.
        """
        flags: List[ValidationFlag] = []

        for i, item in enumerate(po_data.line_items):
            if None in (item.quantity, item.unit_price, item.line_total):
                continue  # Can't check — skip

            computed = item.quantity * item.unit_price  # type: ignore[operator]
            declared = item.line_total                  # type: ignore[arg-type]

            if not _within_tolerance(computed, declared):
                label = (
                    item.description[:40]
                    if item.description
                    else f"line item {i + 1}"
                )
                flags.append(ValidationFlag(
                    rule=f"line_item.total_mismatch:{i}",
                    message=(
                        f"'{label}': "
                        f"quantity({item.quantity}) × unit_price({item.unit_price:.2f}) "
                        f"= {computed:.2f}, but line_total is {declared:.2f}. "
                        f"Difference: {abs(computed - declared):.2f}. "
                        f"May indicate a discount, surcharge, or extraction error."
                    ),
                    severity="warning",
                    affected_fields=["line_items"],
                ))

        # Warn if no line items at all — unusual for a PO
        if not po_data.line_items:
            flags.append(ValidationFlag(
                rule="line_items.empty",
                message=(
                    "No line items were extracted. "
                    "Most POs contain a line item table — verify the document."
                ),
                severity="warning",
                affected_fields=["line_items"],
            ))

        return flags

    # ------------------------------------------------------------------
    # Rule 4: Date ordering
    # ------------------------------------------------------------------

    @staticmethod
    def _check_date_ordering(po_data: POData) -> List[ValidationFlag]:
        """
        Validate date fields individually and against each other.

        Checks:
          - po_date and due_date are parseable ISO 8601 dates
          - due_date is not before po_date
          - po_date is not more than 1 year in the future (likely extraction error)
          - po_date is not more than 10 years in the past (likely extraction error)
        """
        flags: List[ValidationFlag] = []
        today = date.today()

        po_date = _parse_date(po_data.po_date)
        due_date = _parse_date(po_data.due_date)

        # Unparseable date strings
        if po_data.po_date and po_date is None:
            flags.append(ValidationFlag(
                rule="date.po_date_invalid_format",
                message=(
                    f"PO Date '{po_data.po_date}' could not be parsed as a date. "
                    f"Expected ISO 8601 format: YYYY-MM-DD."
                ),
                severity="warning",
                affected_fields=["po_date"],
            ))

        if po_data.due_date and due_date is None:
            flags.append(ValidationFlag(
                rule="date.due_date_invalid_format",
                message=(
                    f"Due Date '{po_data.due_date}' could not be parsed as a date. "
                    f"Expected ISO 8601 format: YYYY-MM-DD."
                ),
                severity="warning",
                affected_fields=["due_date"],
            ))

        # Due date before PO date
        if po_date and due_date and due_date < po_date:
            flags.append(ValidationFlag(
                rule="date.due_before_po",
                message=(
                    f"Due date ({po_data.due_date}) is before PO date ({po_data.po_date}). "
                    f"This is likely an extraction error — verify both dates."
                ),
                severity="error",
                affected_fields=["po_date", "due_date"],
            ))

        # PO date too far in the future
        if po_date:
            years_ahead = (po_date - today).days / 365
            if years_ahead > 1:
                flags.append(ValidationFlag(
                    rule="date.po_date_far_future",
                    message=(
                        f"PO Date ({po_data.po_date}) is more than 1 year in the future. "
                        f"This may be a date extraction error (e.g. year misread)."
                    ),
                    severity="warning",
                    affected_fields=["po_date"],
                ))

            years_past = (today - po_date).days / 365
            if years_past > 10:
                flags.append(ValidationFlag(
                    rule="date.po_date_far_past",
                    message=(
                        f"PO Date ({po_data.po_date}) is more than 10 years in the past. "
                        f"Verify this is not a date extraction error."
                    ),
                    severity="warning",
                    affected_fields=["po_date"],
                ))

        return flags

    # ------------------------------------------------------------------
    # Rule 5: Currency consistency
    # ------------------------------------------------------------------

    @staticmethod
    def _check_currency_consistency(po_data: POData) -> List[ValidationFlag]:
        """
        Collect all currency values across the document and warn on mismatches.

        Mixed currencies can be legitimate (international POs, multi-currency
        contracts) but are far more often extraction errors. A warning surfaces
        it for human confirmation without blocking the result.
        """
        flags: List[ValidationFlag] = []

        currencies: List[str] = []

        if po_data.currency:
            currencies.append(("document", po_data.currency.upper()))
        if po_data.totals and po_data.totals.currency:
            currencies.append(("totals", po_data.totals.currency.upper()))
        for i, item in enumerate(po_data.line_items):
            if item.currency:
                currencies.append((f"line_items[{i}]", item.currency.upper()))

        unique_currencies = set(c for _, c in currencies)
        if len(unique_currencies) > 1:
            summary = ", ".join(f"{loc}={cur}" for loc, cur in currencies)
            flags.append(ValidationFlag(
                rule="currency.inconsistent",
                message=(
                    f"Multiple currencies detected across the document: {summary}. "
                    f"This may indicate an extraction error or a multi-currency PO. "
                    f"Verify before processing."
                ),
                severity="warning",
                affected_fields=["currency", "totals.currency"],
            ))

        return flags

    # ------------------------------------------------------------------
    # Rule 6: Format checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_formats(po_data: POData) -> List[ValidationFlag]:
        """
        Validate field formats that have known patterns.

        Currently checks:
          - Email addresses (vendor and buyer) against a simple RFC-ish regex
          - PO number is not just whitespace or special characters

        Not checking:
          - Phone number format — too varied internationally
          - Tax ID format — varies enormously by country
          - Address format — unstructured, no universal format
        """
        flags: List[ValidationFlag] = []

        # Email validation
        email_fields = []
        if po_data.vendor and po_data.vendor.email:
            email_fields.append(("vendor.email", po_data.vendor.email))
        if po_data.buyer and po_data.buyer.email:
            email_fields.append(("buyer.email", po_data.buyer.email))

        for field_path, email in email_fields:
            if not _is_valid_email(email):
                flags.append(ValidationFlag(
                    rule=f"format.invalid_email:{field_path}",
                    message=(
                        f"'{email}' does not appear to be a valid email address. "
                        f"Verify the extracted value."
                    ),
                    severity="warning",
                    affected_fields=[field_path],
                ))

        # PO number: not just symbols/whitespace
        if po_data.po_number:
            stripped = re.sub(r"[^a-zA-Z0-9]", "", po_data.po_number)
            if not stripped:
                flags.append(ValidationFlag(
                    rule="format.po_number_no_alphanumeric",
                    message=(
                        f"PO number '{po_data.po_number}' contains no alphanumeric characters. "
                        f"This is likely an extraction error."
                    ),
                    severity="error",
                    affected_fields=["po_number"],
                ))

        return flags


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _within_tolerance(a: float, b: float) -> bool:
    """
    Return True if |a - b| is within the accepted financial tolerance.

    Tolerance = max(ABS_TOLERANCE, REL_TOLERANCE × max(|a|, |b|))

    Examples at _ABS_TOLERANCE=0.05, _REL_TOLERANCE=0.01:
      a=100.00, b=100.03 → diff=0.03, tol=max(0.05, 1.00)=1.00 → True
      a=100.00, b=102.00 → diff=2.00, tol=max(0.05, 1.00)=1.00 → False
      a=0.10,   b=0.14   → diff=0.04, tol=max(0.05, 0.001)=0.05 → True
      a=0.10,   b=0.16   → diff=0.06, tol=max(0.05, 0.001)=0.05 → False
    """
    diff = abs(a - b)
    tolerance = max(_ABS_TOLERANCE, _REL_TOLERANCE * max(abs(a), abs(b), 1.0))
    return diff <= tolerance


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """
    Parse an ISO 8601 date string to a date object.
    Returns None if the string is None, empty, or unparseable.
    """
    if not date_str:
        return None
    # Try ISO 8601 first, then common variants
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return date.fromisoformat(date_str) if fmt == "%Y-%m-%d" \
                else date(*[int(x) for x in date_str.split("/")])
        except (ValueError, AttributeError):
            continue
    return None


def _is_valid_email(email: str) -> bool:
    """
    Basic email format validation. Not RFC 5322 compliant — pragmatic check
    for the most common extraction errors (missing @, no domain, etc.)
    """
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))