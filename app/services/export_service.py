"""
app/services/export_service.py

Local export generation and ERP webhook dispatch (architecture decision #9).

Design principles from decision #9:
  - Files generated locally — never uploaded to a third party (GDPR compliance)
  - Webhook is the only ERP coupling — change the URL, change the ERP
  - Webhook payload is intentionally minimal — clean POData only, no pipeline internals
  - Export is gated on is_ready_for_export — bad data never leaves the system

Three export formats:
  JSON — full POData, optional _metadata block with confidence scores
  CSV  — denormalised: one row per line item, header fields repeated (ERP-friendly)
  XML  — structured with Header/LineItems/Totals sections (EDI-friendly)

All formats use only Python stdlib — no new dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import httpx

from app.config import get_settings
from app.models.po_models import (
    ExportFormat,
    ExportRequest,
    ExtractionResult,
    LineItem,
    WebhookPayload,
)

logger = logging.getLogger(__name__)


class ExportService:
    """
    Generates export files and fires ERP webhooks.

    Usage in routes:
        result = document_service.get_result(request.result_id)
        file_path = await export_service.generate_export(result, export_request)
        await export_service.fire_webhook(result)   # optional, if configured
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._export_dir = Path(self._settings.export_dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Export file generation
    # ------------------------------------------------------------------

    async def generate_export(
        self,
        result: ExtractionResult,
        request: ExportRequest,
    ) -> Path:
        """
        Generate an export file for the given result and return its path.

        Gate: refuses to export if result.is_ready_for_export is False.
        This enforces the human review loop — flagged documents don't leave
        the system until a reviewer clears them.

        Files are written to:
          {export_dir}/{result_id}/po_export.{format}

        One directory per result so multiple format exports don't collide
        and cleanup is trivial (delete the result directory).

        Raises:
          ValueError — result not export-ready, or unsupported format.
        """
        if not result.is_ready_for_export:
            pending = result.fields_flagged_for_review
            errors = [f for f in result.validation_flags if f.severity == "error"]
            raise ValueError(
                f"Result '{result.result_id}' is not ready for export. "
                f"Pending review fields: {pending}. "
                f"Validation errors: {[e.rule for e in errors]}. "
                f"Resolve all flags before exporting."
            )

        # Build output directory and file path
        out_dir = self._export_dir / result.result_id
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"po_export.{request.format.value}"

        logger.info(
            "Generating %s export for result %s → %s",
            request.format.value.upper(), result.result_id, file_path,
        )

        # Dispatch to format-specific generator
        if request.format == ExportFormat.JSON:
            content = self._export_json(result, request.include_metadata)
        elif request.format == ExportFormat.CSV:
            content = self._export_csv(result)
        elif request.format == ExportFormat.XML:
            content = self._export_xml(result)
        else:
            raise ValueError(f"Unsupported export format: {request.format}")

        file_path.write_bytes(content)
        logger.info(
            "Export written: %s (%d bytes)", file_path, len(content)
        )
        return file_path

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    @staticmethod
    def _export_json(
        result: ExtractionResult,
        include_metadata: bool,
    ) -> bytes:
        """
        Export as JSON.

        Default (include_metadata=False):
          Clean POData only — what the ERP or downstream system needs.

        With include_metadata=True:
          Adds a _metadata block with confidence scores and pipeline provenance.
          Useful for debugging, auditing, or feeding a data quality dashboard.
          The leading underscore signals "not for ERP consumption."
        """
        output: dict = result.po_data.model_dump(exclude_none=True)

        if include_metadata:
            output["_metadata"] = {
                "result_id": result.result_id,
                "document_id": result.document_id,
                "overall_confidence": result.overall_confidence,
                "fields_flagged_for_review": result.fields_flagged_for_review,
                "primary_model": result.primary_model.value,
                "fallback_triggered": result.fallback_triggered,
                "fallback_fields": result.fallback_fields,
                "processing_duration_ms": result.processing_duration_ms,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "field_confidence": {
                    path: round(ext.confidence, 3)
                    for path, ext in result.field_extractions.items()
                    if ext.value is not None
                },
            }

        return json.dumps(output, indent=2, ensure_ascii=False).encode("utf-8")

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    @staticmethod
    def _export_csv(result: ExtractionResult) -> bytes:
        """
        Export as denormalised CSV — one row per line item.

        Denormalised means PO header fields (po_number, vendor_name, etc.)
        repeat on every row. This is the format most ERP CSV importers expect
        because they process rows independently without a header/detail join.

        If there are no line items, one row is written with header data only
        and empty line-item columns — so the file always has at least one data row.
        """
        po = result.po_data

        # Header columns — fixed across all rows
        header_fields = [
            "po_number", "po_date", "due_date", "payment_terms",
            "delivery_terms", "currency",
            "vendor_name", "vendor_address", "vendor_city", "vendor_country",
            "vendor_email", "vendor_phone", "vendor_tax_id",
            "buyer_name", "buyer_department", "buyer_contact_person",
            "total_subtotal", "total_tax", "total_shipping",
            "total_discount", "total_grand",
        ]
        # Line item columns — per-row
        line_item_fields = [
            "line_number", "line_description", "part_number",
            "quantity", "unit", "unit_price", "line_total",
            "line_currency", "tax_rate",
        ]
        all_columns = header_fields + line_item_fields

        # Build the header row dict (constant across all rows)
        def _str(v) -> str:
            return "" if v is None else str(v)

        header_row = {
            "po_number":           _str(po.po_number),
            "po_date":             _str(po.po_date),
            "due_date":            _str(po.due_date),
            "payment_terms":       _str(po.payment_terms),
            "delivery_terms":      _str(po.delivery_terms),
            "currency":            _str(po.currency),
            "vendor_name":         _str(po.vendor.name if po.vendor else None),
            "vendor_address":      _str(po.vendor.address if po.vendor else None),
            "vendor_city":         _str(po.vendor.city if po.vendor else None),
            "vendor_country":      _str(po.vendor.country if po.vendor else None),
            "vendor_email":        _str(po.vendor.email if po.vendor else None),
            "vendor_phone":        _str(po.vendor.phone if po.vendor else None),
            "vendor_tax_id":       _str(po.vendor.tax_id if po.vendor else None),
            "buyer_name":          _str(po.buyer.name if po.buyer else None),
            "buyer_department":    _str(po.buyer.department if po.buyer else None),
            "buyer_contact_person":_str(po.buyer.contact_person if po.buyer else None),
            "total_subtotal":      _str(po.totals.subtotal if po.totals else None),
            "total_tax":           _str(po.totals.tax_amount if po.totals else None),
            "total_shipping":      _str(po.totals.shipping if po.totals else None),
            "total_discount":      _str(po.totals.discount if po.totals else None),
            "total_grand":         _str(po.totals.grand_total if po.totals else None),
        }

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=all_columns, lineterminator="\r\n")
        writer.writeheader()

        if po.line_items:
            for i, item in enumerate(po.line_items):
                row = dict(header_row)
                row.update({
                    "line_number":      _str(item.line_number or i + 1),
                    "line_description": _str(item.description),
                    "part_number":      _str(item.part_number),
                    "quantity":         _str(item.quantity),
                    "unit":             _str(item.unit),
                    "unit_price":       _str(item.unit_price),
                    "line_total":       _str(item.line_total),
                    "line_currency":    _str(item.currency),
                    "tax_rate":         _str(item.tax_rate),
                })
                writer.writerow(row)
        else:
            # No line items — write one header-only row with empty line columns
            row = dict(header_row)
            row.update({f: "" for f in line_item_fields})
            writer.writerow(row)

        return buf.getvalue().encode("utf-8-sig")  # BOM for Excel compatibility

    # ------------------------------------------------------------------
    # XML export
    # ------------------------------------------------------------------

    @staticmethod
    def _export_xml(result: ExtractionResult) -> bytes:
        """
        Export as XML with Header / LineItems / Totals structure.

        No external dependencies — uses stdlib xml.etree.ElementTree.
        Structure is generic enough for most ERP XML importers.
        Skips None values — absent fields produce no element rather than
        an empty tag, keeping the output clean.
        """
        po = result.po_data

        def _sub(parent: ET.Element, tag: str, value) -> Optional[ET.Element]:
            """Add a child element only if value is not None."""
            if value is None:
                return None
            el = ET.SubElement(parent, tag)
            el.text = str(value)
            return el

        root = ET.Element("PurchaseOrder")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("generated_at", datetime.now(timezone.utc).isoformat())
        root.set("result_id", result.result_id)

        # Header section
        header = ET.SubElement(root, "Header")
        _sub(header, "PONumber",      po.po_number)
        _sub(header, "PODate",        po.po_date)
        _sub(header, "DueDate",       po.due_date)
        _sub(header, "PaymentTerms",  po.payment_terms)
        _sub(header, "DeliveryTerms", po.delivery_terms)
        _sub(header, "Currency",      po.currency)

        # Vendor section
        if po.vendor:
            vendor_el = ET.SubElement(header, "Vendor")
            _sub(vendor_el, "Name",       po.vendor.name)
            _sub(vendor_el, "Address",    po.vendor.address)
            _sub(vendor_el, "City",       po.vendor.city)
            _sub(vendor_el, "Country",    po.vendor.country)
            _sub(vendor_el, "Email",      po.vendor.email)
            _sub(vendor_el, "Phone",      po.vendor.phone)
            _sub(vendor_el, "TaxID",      po.vendor.tax_id)
            _sub(vendor_el, "BankAccount",po.vendor.bank_account)

        # Buyer section
        if po.buyer:
            buyer_el = ET.SubElement(header, "Buyer")
            _sub(buyer_el, "Name",          po.buyer.name)
            _sub(buyer_el, "Address",       po.buyer.address)
            _sub(buyer_el, "Department",    po.buyer.department)
            _sub(buyer_el, "ContactPerson", po.buyer.contact_person)
            _sub(buyer_el, "Email",         po.buyer.email)

        # Line items section
        if po.line_items:
            items_el = ET.SubElement(root, "LineItems")
            items_el.set("count", str(len(po.line_items)))
            for i, item in enumerate(po.line_items):
                item_el = ET.SubElement(items_el, "LineItem")
                item_el.set("index", str(i + 1))
                _sub(item_el, "LineNumber",  item.line_number)
                _sub(item_el, "Description", item.description)
                _sub(item_el, "PartNumber",  item.part_number)
                _sub(item_el, "Quantity",    item.quantity)
                _sub(item_el, "Unit",        item.unit)
                _sub(item_el, "UnitPrice",   item.unit_price)
                _sub(item_el, "LineTotal",   item.line_total)
                _sub(item_el, "Currency",    item.currency)
                _sub(item_el, "TaxRate",     item.tax_rate)
                _sub(item_el, "Notes",       item.notes)

        # Totals section
        if po.totals:
            totals_el = ET.SubElement(root, "Totals")
            _sub(totals_el, "Subtotal",   po.totals.subtotal)
            _sub(totals_el, "TaxAmount",  po.totals.tax_amount)
            _sub(totals_el, "Shipping",   po.totals.shipping)
            _sub(totals_el, "Discount",   po.totals.discount)
            _sub(totals_el, "GrandTotal", po.totals.grand_total)
            _sub(totals_el, "Currency",   po.totals.currency)

        # Pretty-print with indentation (Python 3.9+)
        ET.indent(root, space="  ")
        xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        return xml_bytes

    # ------------------------------------------------------------------
    # ERP webhook dispatch
    # ------------------------------------------------------------------

    async def fire_webhook(self, result: ExtractionResult) -> bool:
        """
        POST a WebhookPayload to the configured ERP webhook URL.

        Only fires if:
          - WEBHOOK_URL is configured in settings
          - result.is_ready_for_export is True (no pending flags)

        Returns True if the webhook was accepted (2xx response), False otherwise.
        Does not raise — webhook failure is logged but never crashes the pipeline.
        The client still gets their export file regardless of webhook status.

        HMAC signing:
          If WEBHOOK_SECRET is set, adds X-Signature: sha256=<hex> header.
          The receiver validates this to confirm the payload came from this system.
          Loosely coupled — if the receiver doesn't check the header, it still works.
        """
        if not self._settings.webhooks_enabled:
            logger.debug("Webhook not configured — skipping.")
            return False

        if not result.is_ready_for_export:
            logger.warning(
                "Webhook not fired for result %s — result is not export-ready.",
                result.result_id,
            )
            return False

        payload = WebhookPayload(
            result_id=result.result_id,
            document_id=result.document_id,
            po_data=result.po_data,
            ready_for_processing=result.is_ready_for_export,
        )

        payload_bytes = payload.model_dump_json(exclude_none=True).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "X-PO-Intelligence-Version": "0.1.0",
            "X-Result-ID": result.result_id,
        }

        if self._settings.webhook_secret:
            headers["X-Signature"] = self._sign_payload(
                payload_bytes, self._settings.webhook_secret
            )

        try:
            async with httpx.AsyncClient(
                timeout=self._settings.webhook_timeout_seconds
            ) as client:
                response = await client.post(
                    self._settings.webhook_url,
                    content=payload_bytes,
                    headers=headers,
                )

            if response.status_code < 300:
                logger.info(
                    "Webhook delivered: result_id=%s → %s (HTTP %d)",
                    result.result_id,
                    self._settings.webhook_url,
                    response.status_code,
                )
                return True
            else:
                logger.warning(
                    "Webhook rejected: result_id=%s, HTTP %d, body=%s",
                    result.result_id,
                    response.status_code,
                    response.text[:200],
                )
                return False

        except httpx.TimeoutException:
            logger.error(
                "Webhook timed out after %ds: result_id=%s, url=%s",
                self._settings.webhook_timeout_seconds,
                result.result_id,
                self._settings.webhook_url,
            )
            return False
        except httpx.ConnectError as e:
            logger.error(
                "Webhook connection failed: result_id=%s, url=%s, error=%s",
                result.result_id, self._settings.webhook_url, e,
            )
            return False
        except Exception as e:
            logger.error(
                "Webhook unexpected error: result_id=%s, error=%s",
                result.result_id, e,
            )
            return False

    # ------------------------------------------------------------------
    # HMAC signing
    # ------------------------------------------------------------------

    @staticmethod
    def _sign_payload(payload_bytes: bytes, secret: str) -> str:
        """
        Generate HMAC-SHA256 signature for webhook payload verification.

        Format: "sha256=<hex_digest>" — same convention as GitHub webhooks,
        so receivers can use standard webhook signature verification libraries.

        The receiver validates by computing the same HMAC with their copy of
        WEBHOOK_SECRET and comparing to the X-Signature header value.
        Use hmac.compare_digest() on the receiver side — not == — to prevent
        timing attacks.
        """
        digest = hmac.new(
            secret.encode("utf-8"),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={digest}"