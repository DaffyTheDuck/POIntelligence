"""
app/models/po_models.py

Single source of truth for all data shapes in the PO Intelligence pipeline.
Every service, provider, and API route imports from here — never the reverse.

Design principles:
  - Clean domain data (what the document contains) is kept separate from
    extraction metadata (how confident we are, which model produced it).
  - Per-field confidence tracking is a first-class citizen, not an afterthought.
  - Architecture decisions are encoded directly into the type system so the
    compiler catches violations before runtime does.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants — architecture decision #3
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.75  # Below this → escalate to Claude
DISAGREEMENT_THRESHOLD = 0.15  # |local_score - claude_score| > this → flag


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """
    Lifecycle states for async email-ingested documents (architecture decision #1).
    Upload flow never enters PENDING — it goes straight to synchronous processing.
    """
    PENDING = "pending"        # Queued, waiting for a Celery worker
    PROCESSING = "processing"  # Worker has picked it up, GPU running inference
    COMPLETE = "complete"      # Extraction finished, result available
    FAILED = "failed"          # Unrecoverable error — client should retry or escalate


class ModelSource(str, Enum):
    """
    Which model produced a given field value.
    Tracked per-field so the audit trail is granular, not document-level.
    """
    LOCAL  = "local"   # llava-phi3 via Ollama on Linux GPU machine
    CLAUDE = "claude"  # Claude API (kept for backward compatibility)
    GROQ   = "groq"    # Groq API (confidence-based fallback, decision #3)
    HUMAN  = "human"   # Human correction applied (decision #5)


class OCRMatchMethod(str, Enum):
    """
    Which layer of the three-layer OCR matching resolved a bounding box (decision #8).
    Stored so you can measure how often you're falling back to expensive LLM hints.
    """
    EXACT = "exact"          # Layer 1: exact string match
    FUZZY = "fuzzy"          # Layer 2: normalised fuzzy match
    LLM_HINT = "llm_hint"    # Layer 3: LLM spatial hint (last resort)
    UNRESOLVED = "unresolved"  # No bounding box found — coordinates will be None


class ReviewReason(str, Enum):
    """Why a field was flagged for human-in-the-loop review (decision #4)."""
    LOW_CONFIDENCE = "low_confidence"    # Single model below CONFIDENCE_THRESHOLD
    MODEL_DISAGREEMENT = "model_disagreement"  # Local vs Claude values differ
    VALIDATION_FAILURE = "validation_failure"  # Business rule violated (e.g. totals mismatch)
    MISSING_REQUIRED = "missing_required"  # Required field not found in document


class ExportFormat(str, Enum):
    """Supported export formats (decision #9 — generated locally for GDPR compliance)."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"     # Common ERP interchange format


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """
    Pixel-coordinate bounding box from surya OCR (architecture decision #7).

    Coordinate system: origin top-left, x grows right, y grows down.
    surya returns absolute pixel coordinates on the original image resolution —
    store them exactly, let the frontend scale to viewport.

    Why surya owns this and not the LLM:
      In a real previous project, asking the LLM for bounding boxes produced
      hallucinated coordinates. surya does layout detection reliably; the LLM
      does semantic understanding reliably. Keep them in their lanes.
    """
    x0: float = Field(..., description="Left edge in pixels")
    y0: float = Field(..., description="Top edge in pixels")
    x1: float = Field(..., description="Right edge in pixels")
    y1: float = Field(..., description="Bottom edge in pixels")
    page: int = Field(default=1, ge=1, description="1-indexed page number")
    ocr_match_method: OCRMatchMethod = Field(
        default=OCRMatchMethod.UNRESOLVED,
        description="Which OCR matching layer resolved this box"
    )

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x1 <= self.x0:
            raise ValueError(f"x1 ({self.x1}) must be greater than x0 ({self.x0})")
        if self.y1 <= self.y0:
            raise ValueError(f"y1 ({self.y1}) must be greater than y0 ({self.y0})")
        return self


class FieldExtraction(BaseModel):
    """
    A single extracted field, wrapped with all the metadata the pipeline needs.

    This is the core primitive that flows through the entire pipeline. Rather than
    storing just `vendor_name: "Acme Corp"`, we store the value alongside confidence,
    provenance, bounding box, and review status. This lets every downstream service
    make decisions without re-running inference.

    Architecture decisions encoded here:
      - Decision #3: confidence < CONFIDENCE_THRESHOLD → field gets escalated
      - Decision #4: flagged_for_review + review_reason capture disagreement
      - Decision #5: source_model tracks provenance for active learning
      - Decision #7: bounding_box comes from surya, not the LLM
    """
    field_name: str = Field(..., description="Canonical field name, e.g. 'vendor_name'")
    value: Optional[Any] = Field(default=None, description="Extracted value, None if not found")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score [0, 1]. Below 0.75 triggers Claude fallback."
    )
    source_model: ModelSource = Field(
        ...,
        description="Which model produced this value"
    )
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Pixel location in the source document. None if OCR matching failed."
    )
    flagged_for_review: bool = Field(
        default=False,
        description="True → human must confirm before this value is trusted"
    )
    review_reason: Optional[ReviewReason] = Field(
        default=None,
        description="Why the field was flagged. Required when flagged_for_review=True."
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="The raw OCR text the model was given for this region"
    )

    @model_validator(mode="after")
    def review_reason_required_when_flagged(self) -> "FieldExtraction":
        if self.flagged_for_review and self.review_reason is None:
            raise ValueError("review_reason must be set when flagged_for_review=True")
        return self


# ---------------------------------------------------------------------------
# Domain models — the clean PO data shape (no metadata here)
# ---------------------------------------------------------------------------


class VendorInfo(BaseModel):
    """
    Vendor/supplier details extracted from the PO header.
    All fields optional because not every PO includes every field.
    Downstream validation (validation_service.py) decides what's required.
    """
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    tax_id: Optional[str] = Field(default=None, description="VAT / GST / EIN number")
    bank_account: Optional[str] = None


class BuyerInfo(BaseModel):
    """Buyer/company details — the entity issuing the PO."""
    name: Optional[str] = None
    address: Optional[str] = None
    department: Optional[str] = None
    contact_person: Optional[str] = None
    email: Optional[str] = None


class LineItem(BaseModel):
    """
    A single line item from the PO table.

    line_total should equal quantity * unit_price, but we store all three
    independently because real-world POs often have rounding errors or
    discount columns — the validation service reconciles them, not this model.
    """
    line_number: Optional[int] = Field(default=None, description="Row number in the PO table")
    description: str
    part_number: Optional[str] = None
    quantity: Optional[float] = Field(default=None, ge=0)
    unit: Optional[str] = Field(default=None, description="e.g. 'pcs', 'kg', 'hours'")
    unit_price: Optional[float] = Field(default=None, ge=0)
    currency: Optional[str] = Field(default=None, min_length=3, max_length=3)
    line_total: Optional[float] = Field(default=None, ge=0)
    tax_rate: Optional[float] = Field(default=None, ge=0, le=1, description="e.g. 0.18 for 18%")
    notes: Optional[str] = None


class POTotals(BaseModel):
    """
    Financial summary section of the PO.
    Stored flat so the validation service can run total reconciliation easily.
    """
    subtotal: Optional[float] = Field(default=None, ge=0)
    tax_amount: Optional[float] = Field(default=None, ge=0)
    shipping: Optional[float] = Field(default=None, ge=0)
    discount: Optional[float] = Field(default=None, ge=0)
    grand_total: Optional[float] = Field(default=None, ge=0)
    currency: Optional[str] = Field(default=None, min_length=3, max_length=3)


class POData(BaseModel):
    """
    The clean, structured data extracted from a purchase order.

    This model contains ONLY domain data — no confidence scores, no bounding boxes,
    no model provenance. It's what you'd put in a database or send to an ERP.

    Compare with ExtractionResult below, which wraps this with full field-level metadata.
    The separation matters: downstream consumers (export, webhook) only want POData.
    The pipeline internals (router, validation) need ExtractionResult.
    """
    po_number: Optional[str] = None
    po_date: Optional[str] = Field(default=None, description="ISO 8601 date string")
    due_date: Optional[str] = Field(default=None, description="Payment or delivery due date")
    payment_terms: Optional[str] = Field(default=None, description="e.g. 'Net 30'")
    delivery_terms: Optional[str] = Field(default=None, description="e.g. 'FOB Destination'")
    currency: Optional[str] = Field(default=None, min_length=3, max_length=3)

    vendor: Optional[VendorInfo] = None
    buyer: Optional[BuyerInfo] = None
    line_items: List[LineItem] = Field(default_factory=list)
    totals: Optional[POTotals] = None

    notes: Optional[str] = Field(default=None, description="Free-text notes or special instructions")
    document_language: Optional[str] = Field(
        default=None,
        description="ISO 639-1 language code detected in the document"
    )

    @field_validator("po_date", "due_date", mode="before")
    @classmethod
    def coerce_date_to_string(cls, v: Any) -> Optional[str]:
        """Accept datetime objects and convert to ISO string for uniform storage."""
        if isinstance(v, datetime):
            return v.date().isoformat()
        return v


# ---------------------------------------------------------------------------
# Extraction result — the full pipeline output
# ---------------------------------------------------------------------------


class ValidationFlag(BaseModel):
    """A single business-rule validation failure on the extracted document."""
    rule: str = Field(..., description="e.g. 'line_items_sum_matches_subtotal'")
    message: str = Field(..., description="Human-readable explanation")
    severity: str = Field(default="warning", description="'error' or 'warning'")
    affected_fields: List[str] = Field(
        default_factory=list,
        description="Field names involved in this validation failure"
    )


class ExtractionResult(BaseModel):
    """
    The complete output of one pass through the extraction pipeline.

    This is what the pipeline services pass between each other internally.
    It contains:
      - po_data: the clean domain model (POData)
      - field_extractions: per-field metadata dict, keyed by field name
      - overall_confidence: document-level score (min of all field scores)
      - pipeline metadata: timing, model routing decisions, flags

    The frontend receives this full object so it can render confidence indicators,
    highlight low-confidence fields, and display bounding boxes for verification.
    """
    result_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for this extraction result"
    )
    document_id: str = Field(..., description="Links back to the source document")
    po_data: POData = Field(..., description="The extracted purchase order data")

    # Per-field extractions, keyed by field path e.g. "vendor.name", "totals.grand_total"
    field_extractions: Dict[str, FieldExtraction] = Field(
        default_factory=dict,
        description="Field-level confidence and metadata, keyed by field path"
    )

    # Document-level quality metrics
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence across all extracted fields (weakest-link score)"
    )
    fields_flagged_for_review: List[str] = Field(
        default_factory=list,
        description="Field names that require human confirmation before use"
    )
    validation_flags: List[ValidationFlag] = Field(
        default_factory=list,
        description="Business rule violations found by validation_service"
    )

    # Pipeline routing metadata (architecture decisions #2 and #3)
    primary_model: ModelSource = Field(
        ...,
        description="Which model was used first (auto-routed based on document quality)"
    )
    fallback_triggered: bool = Field(
        default=False,
        description="True if Claude was called as fallback (confidence was below threshold)"
    )
    fallback_fields: List[str] = Field(
        default_factory=list,
        description="Fields that were re-extracted by Claude after local model underperformed"
    )

    # Timing
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    processing_duration_ms: Optional[int] = Field(
        default=None,
        description="Wall-clock time from start to completion in milliseconds"
    )

    @property
    def is_ready_for_export(self) -> bool:
        """
        True only when no fields require human review and no error-level validation flags.
        Export service checks this before generating any output file.
        """
        has_unflagged_errors = any(
            f.severity == "error" for f in self.validation_flags
        )
        return not self.fields_flagged_for_review and not has_unflagged_errors


# ---------------------------------------------------------------------------
# Job model — async email ingestion (architecture decision #1)
# ---------------------------------------------------------------------------


class JobRecord(BaseModel):
    """
    Tracks the lifecycle of an async document processing job (email ingestion path).

    Upload path: no JobRecord created. Client blocks until ExtractionResult is returned.
    Email path:  JobRecord created immediately, Celery worker updates it, client polls.

    Stored in Redis alongside the Celery task so the API can return status without
    querying Celery directly (Celery task state is an implementation detail, not an API).
    """
    job_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Stable ID the client polls against — never changes"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Set once the document has been saved to storage"
    )
    status: JobStatus = Field(default=JobStatus.PENDING)
    source_email: Optional[str] = Field(
        default=None,
        description="The email address that sent this document"
    )
    source_subject: Optional[str] = Field(
        default=None,
        description="Email subject line, useful for debugging"
    )
    filename: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = Field(
        default=None,
        description="Set only when status=FAILED. Human-readable description."
    )
    result: Optional[ExtractionResult] = Field(
        default=None,
        description="Populated once status=COMPLETE"
    )

    @property
    def is_terminal(self) -> bool:
        """True when the job will not change state again."""
        return self.status in (JobStatus.COMPLETE, JobStatus.FAILED)


# ---------------------------------------------------------------------------
# Human correction — active learning data (architecture decision #5)
# ---------------------------------------------------------------------------


class HumanCorrection(BaseModel):
    """
    A single field correction submitted by a human reviewer.

    These are gold-labelled training examples. Every correction is stored with full
    provenance so it can be used for:
      1. Immediate confidence recalibration (update this document's field score)
      2. Fine-tuning data (field_name + raw_text + corrected_value triplet)
      3. Threshold tuning (analyse patterns in when/why models were wrong)

    The reviewer_id field is intentionally kept opaque — it's the frontend's
    responsibility to authenticate; the backend just stores what it's given.
    """
    correction_id: str = Field(default_factory=lambda: str(uuid4()))
    result_id: str = Field(..., description="ExtractionResult this correction applies to")
    document_id: str
    field_name: str = Field(..., description="e.g. 'vendor.name', 'totals.grand_total'")

    # The before/after pair — this is the labelled training example
    original_value: Optional[Any] = Field(
        default=None,
        description="What the model extracted (may be None if field was missed entirely)"
    )
    corrected_value: Any = Field(..., description="The ground-truth value from the human")

    # Provenance for learning
    original_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score at time of extraction — tells us how badly model was wrong"
    )
    original_source_model: Optional[ModelSource] = None
    raw_text: Optional[str] = Field(
        default=None,
        description="OCR text that was fed to the model — the input side of the training pair"
    )
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Spatial context — useful for layout-aware fine-tuning"
    )

    # Audit
    reviewer_id: str = Field(..., description="Opaque ID of the human who made this correction")
    corrected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: Optional[str] = Field(
        default=None,
        description="Optional reviewer comment explaining the correction"
    )


# ---------------------------------------------------------------------------
# API request/response models
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    """
    Response to a synchronous document upload (POST /api/v1/documents/upload).
    Returns the full result immediately — no polling needed.
    """
    document_id: str
    result: ExtractionResult
    message: str = "Extraction complete"


class JobStatusResponse(BaseModel):
    """
    Response to a job status poll (GET /api/v1/jobs/{job_id}).
    Frontend polls this on a 3-second interval until is_terminal=True.
    """
    job_id: str
    status: JobStatus
    is_terminal: bool
    result: Optional[ExtractionResult] = Field(
        default=None,
        description="Populated only when status=COMPLETE"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Populated only when status=FAILED"
    )
    created_at: datetime
    updated_at: datetime


class CorrectionRequest(BaseModel):
    """
    Request body for submitting a human correction (POST /api/v1/corrections).
    reviewer_id is injected by the frontend — no auth layer required for the interview demo.
    """
    result_id: str
    field_name: str
    corrected_value: Any
    reviewer_id: str = Field(default="demo_reviewer")
    notes: Optional[str] = None


class CorrectionResponse(BaseModel):
    """Response confirming a correction was saved and the field was recalibrated."""
    correction_id: str
    field_name: str
    new_confidence: float = Field(
        default=1.0,
        description="After human correction, confidence is set to 1.0 — it's ground truth"
    )
    message: str = "Correction saved. Field confidence recalibrated."


# ---------------------------------------------------------------------------
# Export/webhook models (architecture decision #9)
# ---------------------------------------------------------------------------


class ExportRequest(BaseModel):
    """
    Request to generate an export file locally (POST /api/v1/export).
    Files generated on-device for GDPR compliance — never sent to a third party.
    """
    result_id: str
    format: ExportFormat = ExportFormat.JSON
    include_metadata: bool = Field(
        default=False,
        description="If True, field-level confidence scores are included in the export"
    )


class WebhookPayload(BaseModel):
    """
    Payload sent to an ERP or downstream system via webhook (architecture decision #9).

    Intentionally minimal — only clean POData, no pipeline internals.
    The receiving system doesn't care how confident we were; it just needs the data.
    Loose coupling: if the ERP changes, only the webhook URL config changes.
    """
    event: str = Field(default="po.extracted", description="Event type for the receiving system")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result_id: str
    document_id: str
    po_data: POData
    ready_for_processing: bool = Field(
        ...,
        description="False if human review is still pending — ERP should wait"
    )