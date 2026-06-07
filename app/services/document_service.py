"""
app/services/document_service.py

Boundary layer between the HTTP upload endpoint and the extraction pipeline.

Responsibilities:
  1. Validate incoming files (size, MIME type, magic bytes)
  2. Save files to disk under a document_id-scoped path
  3. Call extraction_service and return the result
  4. Store and retrieve ExtractionResult objects
  5. Apply and store human corrections (architecture decision #5)

What this service deliberately does NOT do:
  - No HTTP concerns (no Request, Response, or status code awareness)
  - No job queue interaction (that's email_service + Celery)
  - No business rule validation (that's validation_service)

Storage note:
  ExtractionResults are stored in an in-memory dict for the demo.
  In production, replace _ResultStore with Redis (serialize via model_dump_json)
  or PostgreSQL. The interface is identical — callers use get_result(result_id),
  not the storage implementation.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from app.config import get_settings
from app.models.po_models import (
    CorrectionRequest,
    CorrectionResponse,
    ExtractionResult,
    FieldExtraction,
    HumanCorrection,
    ModelSource,
    ReviewReason,
)
from app.services.extraction_service import ExtractionService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Magic byte signatures for MIME type verification
# (Don't trust the client's declared Content-Type alone)
# ---------------------------------------------------------------------------

_MAGIC_BYTES: Dict[bytes, str] = {
    b"%PDF":              "application/pdf",
    b"\xff\xd8\xff":     "image/jpeg",
    b"\x89PNG\r\n\x1a\n":"image/png",
    b"II*\x00":          "image/tiff",   # Little-endian TIFF
    b"MM\x00*":          "image/tiff",   # Big-endian TIFF
    b"RIFF":             "image/webp",   # WebP (needs further check at byte 8-12)
}

# ---------------------------------------------------------------------------
# In-memory result store (demo only — see module docstring)
# ---------------------------------------------------------------------------


class _ResultStore:
    """
    Thread-safe in-memory store for ExtractionResult objects.

    Keyed by result_id (primary) and document_id (secondary index).
    A threading.Lock is used rather than asyncio.Lock because FastAPI
    route handlers are async but Celery tasks are sync threads — both
    paths write to this store.

    Production replacement: serialize result.model_dump_json() to Redis
    with key "result:{result_id}" and TTL matching redis_job_ttl_seconds.
    """

    def __init__(self) -> None:
        self._by_result_id: Dict[str, ExtractionResult] = {}
        self._by_document_id: Dict[str, str] = {}   # document_id → result_id
        self._corrections: List[HumanCorrection] = []
        self._lock = threading.Lock()

    def save(self, result: ExtractionResult) -> None:
        with self._lock:
            self._by_result_id[result.result_id] = result
            self._by_document_id[result.document_id] = result.result_id

    def get_by_result_id(self, result_id: str) -> Optional[ExtractionResult]:
        with self._lock:
            return self._by_result_id.get(result_id)

    def get_by_document_id(self, document_id: str) -> Optional[ExtractionResult]:
        with self._lock:
            result_id = self._by_document_id.get(document_id)
            return self._by_result_id.get(result_id) if result_id else None

    def update(self, result: ExtractionResult) -> None:
        """Replace a stored result with an updated version (after correction)."""
        with self._lock:
            self._by_result_id[result.result_id] = result

    def save_correction(self, correction: HumanCorrection) -> None:
        with self._lock:
            self._corrections.append(correction)

    def get_corrections(self, result_id: str) -> List[HumanCorrection]:
        with self._lock:
            return [c for c in self._corrections if c.result_id == result_id]


# Module-level singleton — shared across all DocumentService instances
_store = _ResultStore()


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class DocumentService:
    """
    Manages the full lifecycle of an uploaded document.

    Used by:
      - POST /api/v1/documents/upload  (synchronous path — awaits result)
      - Celery task for email ingestion (async path — same extract() call)
    """

    def __init__(
        self,
        extraction_service: Optional[ExtractionService] = None,
    ) -> None:
        self._settings = get_settings()
        self._extraction = extraction_service or ExtractionService()
        self._upload_dir = Path(self._settings.upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point — synchronous upload path
    # ------------------------------------------------------------------

    async def process_upload(
        self,
        file_bytes: bytes,
        filename: str,
        declared_mime_type: str,
    ) -> ExtractionResult:
        """
        Validate, save, extract, and return a result for an uploaded file.

        Raises:
          ValueError  — invalid file (size, type). Routes convert to 4xx.
          RuntimeError — both providers unavailable. Routes convert to 503.
          Any other exception propagates as 500.
        """
        # Step 1: Validate — reject before writing anything to disk
        validated_mime = self._validate_file(
            file_bytes=file_bytes,
            filename=filename,
            declared_mime_type=declared_mime_type,
        )

        # Step 2: Generate a stable document ID and save the file
        document_id = str(uuid4())
        file_path = self._save_file(
            document_id=document_id,
            filename=filename,
            file_bytes=file_bytes,
        )

        logger.info(
            "Document saved: id=%s, path=%s, size=%d bytes, mime=%s",
            document_id, file_path, len(file_bytes), validated_mime,
        )

        # Step 3: Extract — this is where the GPU work happens
        result = await self._extraction.extract(
            file_path=str(file_path),
            document_id=document_id,
            mime_type=validated_mime,
        )

        # Step 4: Store result for later retrieval (corrections, export)
        _store.save(result)

        logger.info(
            "Result stored: result_id=%s, document_id=%s, confidence=%.2f",
            result.result_id, document_id, result.overall_confidence,
        )

        # Step 5: Create a JobRecord so this upload appears in the jobs list
        # Import inside function to avoid circular imports
        try:
            from app.services.email_service import job_store
            from app.models.po_models import JobRecord, JobStatus
            from datetime import datetime, timezone
            upload_job = JobRecord(
                job_id=str(uuid4()),
                document_id=document_id,
                status=JobStatus.COMPLETE,
                filename=filename,
                source_email="upload",   # sentinel — not from email
                result=result,
                completed_at=datetime.now(timezone.utc),
            )
            job_store.save(upload_job)
        except Exception as e:
            logger.warning("Could not register upload in job store: %s", e)

        return result

    # ------------------------------------------------------------------
    # Result retrieval
    # ------------------------------------------------------------------

    def get_result(self, result_id: str) -> Optional[ExtractionResult]:
        """Retrieve a stored ExtractionResult by its result_id."""
        return _store.get_by_result_id(result_id)

    def get_result_by_document(self, document_id: str) -> Optional[ExtractionResult]:
        """Retrieve the ExtractionResult associated with a document_id."""
        return _store.get_by_document_id(document_id)

    # ------------------------------------------------------------------
    # Human corrections — architecture decision #5
    # ------------------------------------------------------------------

    def apply_correction(
        self,
        request: CorrectionRequest,
    ) -> CorrectionResponse:
        """
        Apply a human correction to a stored ExtractionResult.

        What this does:
          1. Retrieve the ExtractionResult by result_id
          2. Find the FieldExtraction for the corrected field
          3. Update: value → corrected value, confidence → 1.0 (ground truth),
             source_model → HUMAN, flagged_for_review → False
          4. Remove the field from fields_flagged_for_review if it was there
          5. Persist the updated result
          6. Store the HumanCorrection record (labelled training data)

        Architecture decision #5:
          HumanCorrection stores the (original_value, corrected_value, raw_text,
          confidence_at_time) tuple. This is the gold-labelled training example.
          Collected corrections feed active learning and fine-tuning pipelines.

        Returns:
          CorrectionResponse confirming the correction and new confidence.

        Raises:
          ValueError if result_id not found or field_name not in result.
        """
        result = _store.get_by_result_id(request.result_id)
        if result is None:
            raise ValueError(
                f"No result found with id '{request.result_id}'. "
                f"The result may have expired or the id is incorrect."
            )

        field_path = request.field_name
        existing_extraction = result.field_extractions.get(field_path)

        # Build the HumanCorrection record (training data, decision #5)
        correction = HumanCorrection(
            correction_id=str(uuid4()),
            result_id=request.result_id,
            document_id=result.document_id,
            field_name=field_path,
            original_value=(
                existing_extraction.value if existing_extraction else None
            ),
            corrected_value=request.corrected_value,
            original_confidence=(
                existing_extraction.confidence if existing_extraction else None
            ),
            original_source_model=(
                existing_extraction.source_model if existing_extraction else None
            ),
            raw_text=(
                existing_extraction.raw_text if existing_extraction else None
            ),
            bounding_box=(
                existing_extraction.bounding_box if existing_extraction else None
            ),
            reviewer_id=request.reviewer_id,
            corrected_at=datetime.now(timezone.utc),
            notes=request.notes,
        )

        # Build the updated FieldExtraction
        updated_extraction = FieldExtraction(
            field_name=field_path,
            value=request.corrected_value,
            confidence=1.0,          # Human-confirmed = ground truth
            source_model=ModelSource.HUMAN,
            bounding_box=(
                existing_extraction.bounding_box if existing_extraction else None
            ),
            flagged_for_review=False,  # Human has reviewed — no longer flagged
            review_reason=None,
            raw_text=(
                existing_extraction.raw_text if existing_extraction else None
            ),
        )

        # Rebuild field_extractions with the correction applied
        updated_extractions = dict(result.field_extractions)
        updated_extractions[field_path] = updated_extraction

        # Remove from flagged list now that a human has confirmed
        updated_flagged = [
            f for f in result.fields_flagged_for_review if f != field_path
        ]

        # Recompute overall_confidence — corrected field is now 1.0
        non_null_confidences = [
            ext.confidence
            for ext in updated_extractions.values()
            if ext.value is not None
        ]
        new_overall = (
            round(min(non_null_confidences), 3) if non_null_confidences else 0.0
        )

        # Rebuild the result with all updates applied
        updated_result = result.model_copy(update={
            "field_extractions": updated_extractions,
            "fields_flagged_for_review": updated_flagged,
            "overall_confidence": new_overall,
        })

        # Persist both the updated result and the correction record
        _store.update(updated_result)
        _store.save_correction(correction)

        logger.info(
            "Correction applied: result_id=%s, field=%s, reviewer=%s, "
            "original=%r → corrected=%r",
            request.result_id,
            field_path,
            request.reviewer_id,
            correction.original_value,
            request.corrected_value,
        )

        return CorrectionResponse(
            correction_id=correction.correction_id,
            field_name=field_path,
            new_confidence=1.0,
        )

    def get_corrections(self, result_id: str) -> List[HumanCorrection]:
        """
        Return all human corrections applied to a given result.
        Used by the active learning pipeline to retrieve training data.
        """
        return _store.get_corrections(result_id)

    # ------------------------------------------------------------------
    # File path helpers
    # ------------------------------------------------------------------

    def get_file_path(self, document_id: str, filename: str) -> Optional[Path]:
        """
        Return the on-disk path of a stored document file.

        Used by routes that serve the original document to the frontend
        (the PDF viewer / image display for bounding box overlay).
        Returns None if the file doesn't exist on disk.
        """
        path = self._upload_dir / document_id / filename
        return path if path.exists() else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_file(
        self,
        file_bytes: bytes,
        filename: str,
        declared_mime_type: str,
    ) -> str:
        """
        Validate file size, declared MIME type, and actual magic bytes.

        Returns the validated MIME type (detected, not just declared).

        Raises ValueError with a user-facing message on any violation.
        """
        # Size check — before any processing
        max_bytes = self._settings.max_upload_size_bytes
        if len(file_bytes) > max_bytes:
            raise ValueError(
                f"File size {len(file_bytes) / 1024 / 1024:.1f}MB exceeds "
                f"the {self._settings.max_upload_size_mb}MB limit."
            )

        if len(file_bytes) == 0:
            raise ValueError("Uploaded file is empty.")

        # Declared MIME type check
        allowed = self._settings.allowed_mime_types
        if declared_mime_type not in allowed:
            raise ValueError(
                f"File type '{declared_mime_type}' is not supported. "
                f"Supported types: {', '.join(allowed)}."
            )

        # Magic byte detection — don't trust declared type alone
        detected_mime = self._detect_mime_type(file_bytes)
        if detected_mime is None:
            raise ValueError(
                f"File '{filename}' does not appear to be a recognised document format. "
                f"Expected PDF or image file."
            )

        if detected_mime not in allowed:
            raise ValueError(
                f"File content detected as '{detected_mime}' which is not supported. "
                f"The declared type was '{declared_mime_type}'."
            )

        # Mismatch warning — client declared wrong type but actual type is valid
        if detected_mime != declared_mime_type:
            logger.warning(
                "MIME type mismatch for '%s': declared=%s, detected=%s. "
                "Using detected type.",
                filename, declared_mime_type, detected_mime,
            )

        return detected_mime

    def _save_file(
        self,
        document_id: str,
        filename: str,
        file_bytes: bytes,
    ) -> Path:
        """
        Save file bytes to {upload_dir}/{document_id}/{filename}.

        Each document gets its own subdirectory so:
          - Cleanup is trivial (delete the directory)
          - No filename collisions between documents
          - Original filename is preserved for display in the UI
        """
        # Sanitise filename — strip path separators that could cause traversal
        safe_filename = Path(filename).name  # Strips any directory components
        if not safe_filename:
            safe_filename = f"document_{document_id}"

        doc_dir = self._upload_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        file_path = doc_dir / safe_filename
        file_path.write_bytes(file_bytes)
        return file_path

    @staticmethod
    def _detect_mime_type(file_bytes: bytes) -> Optional[str]:
        """
        Detect MIME type from magic bytes (first 12 bytes of the file).

        Does not use python-magic to avoid a system library dependency.
        Handles: PDF, JPEG, PNG, TIFF (both endians), WebP.
        """
        header = file_bytes[:12]

        for magic, mime in _MAGIC_BYTES.items():
            if header.startswith(magic):
                # WebP needs an extra check: bytes 8-12 must be "WEBP"
                if mime == "image/webp":
                    if len(file_bytes) >= 12 and file_bytes[8:12] == b"WEBP":
                        return "image/webp"
                    continue  # RIFF but not WEBP — not a supported format
                return mime

        return None