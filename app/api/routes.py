"""
app/api/routes.py

FastAPI router — HTTP surface for the PO Intelligence pipeline.

Seven endpoints:
  POST   /documents/upload          — synchronous upload + extraction
  GET    /jobs/{job_id}             — poll async job status (email path)
  POST   /corrections               — submit human field correction
  POST   /export                    — generate + download export file
  POST   /webhook/trigger           — manually fire ERP webhook
  GET    /documents/{document_id}/file — serve original document for UI overlay
  GET    /health                    — provider liveness check

Design rules enforced here:
  - No business logic — delegate immediately to services
  - Map service exceptions to HTTP status codes explicitly
  - File uploads read fully into memory before passing to service
    (acceptable for the configured max_upload_size_mb limit)
  - All response models declared so FastAPI generates accurate OpenAPI docs

Mounted at /api/v1 in main.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.po_models import (
    ApprovalRecord,
    ApprovalStatus,
    CorrectionRequest,
    CorrectionResponse,
    ExportRequest,
    JobStatusResponse,
    ModelSource,
    UploadResponse,
)
from app.providers.groq_provider import GroqProvider
from app.providers.ollama_provider import OllamaProvider
from app.services.document_service import DocumentService
from app.services.email_service import job_store
from app.services.export_service import ExportService

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Service dependency providers
# ---------------------------------------------------------------------------
# FastAPI's Depends() calls these per-request.
# Constructing services is cheap (just settings lookup), so no caching needed.
# Tests override these with mock implementations via app.dependency_overrides.


def get_document_service() -> DocumentService:
    return DocumentService()


def get_export_service() -> ExportService:
    return ExportService()


# ---------------------------------------------------------------------------
# MIME type map for export file responses
# ---------------------------------------------------------------------------

_EXPORT_MIME: dict = {
    "json": "application/json",
    "csv":  "text/csv; charset=utf-8",
    "xml":  "application/xml",
}


# ===========================================================================
# Document ingestion — synchronous upload path
# ===========================================================================


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    summary="Upload a PO document for synchronous extraction",
    description=(
        "Upload a PDF or image file. The pipeline runs synchronously and returns "
        "the full ExtractionResult in the response. For high-volume ingestion, "
        "use the IMAP email path and poll /jobs/{job_id} instead."
    ),
    status_code=200,
)
async def upload_document(
    file: UploadFile = File(
        ...,
        description="PDF or image file (JPEG, PNG, TIFF, WebP). Max size set by config.",
    ),
    document_service: DocumentService = Depends(get_document_service),
) -> UploadResponse:
    """
    Synchronous upload and extraction endpoint.

    Flow:
      1. Read file bytes from the multipart upload
      2. Delegate to document_service (validates, saves, extracts)
      3. Return UploadResponse with the full ExtractionResult

    HTTP errors:
      400 — invalid file (empty, unsupported format, magic bytes mismatch)
      413 — file exceeds max_upload_size_mb
      503 — both Ollama and Claude are unavailable
    """
    # Read the full file into memory
    # Acceptable given max_upload_size_mb ceiling enforced by document_service
    file_bytes = await file.read()

    # Quick size pre-check before hitting document_service
    # (document_service also checks, but this gives a cleaner 413 response)
    settings = get_settings()
    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size {len(file_bytes) / 1024 / 1024:.1f}MB exceeds "
                f"the {settings.max_upload_size_mb}MB limit."
            ),
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    declared_mime = file.content_type or "application/octet-stream"
    filename = file.filename or "document"

    logger.info(
        "Upload received: filename='%s', mime='%s', size=%d bytes",
        filename, declared_mime, len(file_bytes),
    )

    try:
        result = await document_service.process_upload(
            file_bytes=file_bytes,
            filename=filename,
            declared_mime_type=declared_mime,
        )
    except ValueError as e:
        # document_service raises ValueError for invalid files
        # Determine whether it's a type error (415) or other validation (400)
        msg = str(e)
        status = 415 if "type" in msg.lower() or "format" in msg.lower() else 400
        raise HTTPException(status_code=status, detail=msg)
    except RuntimeError as e:
        # Both providers unavailable
        raise HTTPException(status_code=503, detail=str(e))

    return UploadResponse(
        document_id=result.document_id,
        result=result,
    )


@router.get(
    "/jobs",
    summary="List all recent documents (uploads + email ingestion)",
    description="Returns all processed documents sorted by creation time descending. Used by the Jobs dashboard.",
)
async def list_jobs() -> JSONResponse:
    """
    Return all jobs from both upload and email ingestion paths.
    Upload jobs have source='upload', email jobs have source='email'.
    """
    from app.services.email_service import job_store
    from app.models.po_models import JobRecord

    all_jobs: list[JobRecord] = []

    if job_store._redis:
        try:
            keys = job_store._redis.keys("job:*")
            for key in keys:
                data = job_store._redis.get(key)
                if data:
                    all_jobs.append(JobRecord.model_validate_json(data))
        except Exception as e:
            logger.warning("Redis list failed: %s — using in-memory", e)

    if not all_jobs:
        with job_store._lock:
            for data in job_store._fallback.values():
                try:
                    all_jobs.append(JobRecord.model_validate_json(data))
                except Exception:
                    pass

    all_jobs.sort(key=lambda j: j.created_at, reverse=True)

    return JSONResponse([
        {
            "job_id":        j.job_id,
            "status":        j.status.value if hasattr(j.status, 'value') else j.status,
            "filename":      j.filename or "document",
            "source":        "upload" if j.source_email == "upload" else "email",
            "source_email":  j.source_email if j.source_email != "upload" else None,
            "confidence":    round(j.result.overall_confidence, 3) if j.result else None,
            "flagged":       len(j.result.fields_flagged_for_review) if j.result else None,
            "duration_ms":   j.result.processing_duration_ms if j.result else None,
            "primary_model": (
                j.result.primary_model.value
                if j.result and hasattr(j.result.primary_model, 'value')
                else str(j.result.primary_model) if j.result else None
            ),
            "fallback":      j.result.fallback_triggered if j.result else None,
            "created_at":    j.created_at.isoformat(),
            "error":         j.error_message,
            "result_id":     j.result.result_id if j.result else None,
            "document_id":   j.document_id,
        }
        for j in all_jobs[:100]
    ])


# ---------------------------------------------------------------------------
# Job status polling — async email ingestion path
# ---------------------------------------------------------------------------


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll the status of an async extraction job",
    description=(
        "Poll this endpoint after an email-ingested document is queued. "
        "Returns the current job status. When status='complete', the full "
        "ExtractionResult is included in the response. "
        "Recommended polling interval: 3 seconds."
    ),
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Return the current status of an async extraction job.

    The job_id is returned immediately when an email attachment is queued.
    Poll until is_terminal=True (status is 'complete' or 'failed').

    HTTP errors:
      404 — job_id not found (may have expired after redis_job_ttl_seconds)
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Job '{job_id}' not found. "
                f"It may have expired (TTL: {get_settings().redis_job_ttl_seconds}s) "
                f"or the job_id is incorrect."
            ),
        )

    return job_store.to_status_response(job)


# ===========================================================================
# Human corrections — active learning data (architecture decision #5)
# ===========================================================================


@router.post(
    "/corrections",
    response_model=CorrectionResponse,
    summary="Submit a human correction for an extracted field",
    description=(
        "Correct an extracted field value. The corrected value is stored as "
        "ground-truth labelled data for active learning. The field's confidence "
        "is recalibrated to 1.0 and it is removed from the review queue."
    ),
)
async def submit_correction(
    request: CorrectionRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> CorrectionResponse:
    """
    Apply a human correction to a stored ExtractionResult.

    The correction:
      - Updates the field value to the corrected value
      - Sets confidence to 1.0 (human-confirmed = ground truth)
      - Removes the field from fields_flagged_for_review
      - Stores a HumanCorrection record for active learning

    HTTP errors:
      404 — result_id not found
      400 — other correction errors
    """
    try:
        return document_service.apply_correction(request)
    except ValueError as e:
        msg = str(e)
        status = 404 if "not found" in msg.lower() else 400
        raise HTTPException(status_code=status, detail=msg)


# ===========================================================================
# Approval — explicit reviewer sign-off (human-in-the-loop gate)
# ===========================================================================


class ApproveRequest(BaseModel):
    result_id:   str
    reviewer_id: str = Field(default="reviewer", description="Who is approving")
    notes:       Optional[str] = Field(default=None)


class RejectRequest(BaseModel):
    result_id:   str
    reviewer_id: str = Field(default="reviewer")
    reason:      str = Field(..., description="Required reason for rejection")


class ApproveFieldRequest(BaseModel):
    result_id:   str
    field_name:  str
    reviewer_id: str = Field(default="reviewer")


@router.post(
    "/approve-field",
    summary="Approve an individual field value",
    description="One-click approval for a single field. Sets confidence to 1.0 and marks as human-verified. When all flagged fields are approved, document is ready for final approval.",
)
async def approve_field(
    request: ApproveFieldRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> JSONResponse:
    result = document_service.get_result(request.result_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Result '{request.result_id}' not found.")

    ext = result.field_extractions.get(request.field_name)
    if ext is None:
        raise HTTPException(status_code=404, detail=f"Field '{request.field_name}' not found.")

    # Mark as human-confirmed — same as submitting a correction with unchanged value
    result.field_extractions[request.field_name] = ext.model_copy(update={
        'confidence':         1.0,
        'source_model':       ModelSource.HUMAN,
        'flagged_for_review': False,
        'review_reason':      None,
    })

    # Remove from flagged list
    result.fields_flagged_for_review = [
        f for f in result.fields_flagged_for_review
        if f != request.field_name
    ]

    document_service.update_result(result)

    return JSONResponse({
        "approved":           True,
        "field_name":         request.field_name,
        "fields_remaining":   len(result.fields_flagged_for_review),
        "all_fields_cleared": len(result.fields_flagged_for_review) == 0,
    })


class ApproveAllFieldsRequest(BaseModel):
    result_id:   str
    reviewer_id: str = Field(default="reviewer")


@router.post(
    "/approve-all-fields",
    summary="Bulk-approve all flagged fields",
    description=(
        "One-click approval for every field currently in the review queue. "
        "Each flagged field is marked as human-verified (confidence → 1.0, source → human). "
        "Returns the count of approved fields and the cleared list."
    ),
)
async def approve_all_fields(
    request: ApproveAllFieldsRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> JSONResponse:
    result = document_service.get_result(request.result_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Result '{request.result_id}' not found.")

    flagged = list(result.fields_flagged_for_review)  # snapshot before mutation
    if not flagged:
        return JSONResponse({
            "approved_count": 0,
            "approved_fields": [],
            "message": "No flagged fields to approve.",
        })

    approved_fields = []
    for field_name in flagged:
        ext = result.field_extractions.get(field_name)
        if ext is None:
            continue  # field no longer exists — skip
        result.field_extractions[field_name] = ext.model_copy(update={
            'confidence':         1.0,
            'source_model':       ModelSource.HUMAN,
            'flagged_for_review': False,
            'review_reason':      None,
        })
        approved_fields.append(field_name)

    # Clear the flagged list entirely
    result.fields_flagged_for_review = []

    document_service.update_result(result)

    logger.info(
        "Bulk field approval: result_id=%s reviewer=%s approved=%d fields=%s",
        request.result_id, request.reviewer_id, len(approved_fields), approved_fields,
    )

    return JSONResponse({
        "approved_count":  len(approved_fields),
        "approved_fields": approved_fields,
        "message":         f"{len(approved_fields)} field(s) approved.",
    })


@router.post(
    "/approve",
    summary="Approve a document for export",
)
async def approve_result(
    request: ApproveRequest,
    document_service: DocumentService = Depends(get_document_service),
    export_service:   ExportService   = Depends(get_export_service),
) -> JSONResponse:

    result = document_service.get_result(request.result_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Result '{request.result_id}' not found.")

    result.approval = ApprovalRecord(
        status      = ApprovalStatus.APPROVED,
        reviewer_id = request.reviewer_id,
        approved_at = datetime.now(timezone.utc),
        notes       = request.notes,
    )

    # Persist via the correct update method
    document_service.update_result(result)

    logger.info("Result approved: result_id=%s reviewer=%s", request.result_id, request.reviewer_id)

    # Fire webhook on approval (non-blocking)
    try:
        await export_service.fire_webhook(result)
    except Exception as e:
        logger.warning("Webhook after approval failed: %s", e)

    return JSONResponse({
        "approved":    True,
        "result_id":   result.result_id,
        "reviewer_id": request.reviewer_id,
        "approved_at": result.approval.approved_at.isoformat(),
        "notes":       request.notes,
    })


@router.post(
    "/reject",
    summary="Reject a document — marks it as needing re-review",
)
async def reject_result(
    request: RejectRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> JSONResponse:

    result = document_service.get_result(request.result_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Result '{request.result_id}' not found.")

    result.approval = ApprovalRecord(
        status      = ApprovalStatus.REJECTED,
        reviewer_id = request.reviewer_id,
        approved_at = datetime.now(timezone.utc),
        notes       = request.reason,
    )

    document_service.update_result(result)

    logger.info("Result rejected: result_id=%s reviewer=%s reason=%s",
                request.result_id, request.reviewer_id, request.reason)

    return JSONResponse({
        "rejected":    True,
        "result_id":   result.result_id,
        "reviewer_id": request.reviewer_id,
        "reason":      request.reason,
    })


# ===========================================================================
# Export — local file generation (architecture decision #9)
# ===========================================================================


@router.post(
    "/export",
    summary="Generate and download an export file",
    description=(
        "Generate a JSON, CSV, or XML export of the extracted PO data. "
        "Files are generated locally (GDPR compliance — no third-party upload). "
        "Only available when the result is export-ready (no pending review flags). "
        "If WEBHOOK_URL is configured, the ERP webhook is also fired."
    ),
)
async def export_result(
    request: ExportRequest,
    document_service: DocumentService = Depends(get_document_service),
    export_service: ExportService = Depends(get_export_service),
) -> FileResponse:
    """
    Generate an export file and return it as a download.

    Also fires the ERP webhook if WEBHOOK_URL is configured.
    Webhook failure does not affect the file download response.

    HTTP errors:
      404 — result_id not found
      400 — result not export-ready (pending flags or validation errors)
    """
    result = document_service.get_result(request.result_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Result '{request.result_id}' not found.",
        )

    try:
        file_path = await export_service.generate_export(
            result, request, force=getattr(request, 'force', False)
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Fire webhook (best-effort — failure doesn't affect the file response)
    if get_settings().webhooks_enabled:
        try:
            fired = await export_service.fire_webhook(result)
            if not fired:
                logger.warning(
                    "Webhook not delivered for result %s. "
                    "Use POST /webhook/trigger to retry.",
                    result.result_id,
                )
        except Exception as e:
            logger.error("Webhook error during export: %s", e)

    mime_type = _EXPORT_MIME.get(request.format.value, "application/octet-stream")
    download_name = f"po_{result.po_data.po_number or result.result_id[:8]}.{request.format.value}"

    return FileResponse(
        path=str(file_path),
        media_type=mime_type,
        filename=download_name,
        headers={
            "X-Result-ID": result.result_id,
            "X-Overall-Confidence": str(result.overall_confidence),
        },
    )


# ===========================================================================
# Webhook — manual ERP delivery trigger
# ===========================================================================


@router.post(
    "/webhook/trigger",
    summary="Manually fire the ERP webhook for a result",
    description=(
        "Re-fire the ERP webhook for a result that was already exported. "
        "Useful when the automatic webhook failed or the ERP endpoint was "
        "temporarily unavailable."
    ),
)
async def trigger_webhook(
    result_id: str,
    document_service: DocumentService = Depends(get_document_service),
    export_service: ExportService = Depends(get_export_service),
) -> JSONResponse:
    """
    Manually trigger the ERP webhook for a given result.

    HTTP errors:
      404 — result_id not found
      400 — result not export-ready or WEBHOOK_URL not configured
    """
    if not get_settings().webhooks_enabled:
        raise HTTPException(
            status_code=400,
            detail=(
                "WEBHOOK_URL is not configured. "
                "Set WEBHOOK_URL in your .env file to enable webhook delivery."
            ),
        )

    result = document_service.get_result(result_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Result '{result_id}' not found.",
        )

    if not result.is_ready_for_export:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Result '{result_id}' is not export-ready. "
                f"Pending review fields: {result.fields_flagged_for_review}."
            ),
        )

    fired = await export_service.fire_webhook(result)

    return JSONResponse(
        content={
            "fired": fired,
            "result_id": result_id,
            "webhook_url": get_settings().webhook_url,
            "message": "Webhook delivered." if fired else (
                "Webhook delivery failed. Check server logs for details."
            ),
        },
        status_code=200 if fired else 502,
    )


# ===========================================================================
# Document file serving — original file for UI bounding box overlay
# ===========================================================================


@router.get(
    "/documents/{document_id}/file",
    summary="Serve the original document file",
    description=(
        "Returns the original uploaded document file. "
        "Used by the frontend to render the document with bounding box overlays "
        "for extracted fields. The file is served directly from the local uploads directory."
    ),
)
async def serve_document_file(
    document_id: str,
) -> FileResponse:
    """
    Serve the original document for the frontend document viewer.

    The frontend renders this file and overlays bounding boxes from the
    ExtractionResult.field_extractions[*].bounding_box data.

    HTTP errors:
      404 — document not found or file not on disk
    """
    # Locate the file directly from disk — no store lookup needed.
    # The file exists on disk even if the result was from a previous session
    # or came through the email ingestion path.
    settings = get_settings()
    upload_dir = Path(settings.upload_dir) / document_id

    if not upload_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document file not found on disk for document '{document_id}'.",
        )

    # Serve rasterized PNG first (generated during PDF processing)
    png_path = upload_dir / "page_1.png"
    if png_path.exists():
        return FileResponse(
            path=str(png_path),
            media_type="image/png",
            filename="document.png",
            headers={
                "X-Document-ID": document_id,
                "Cache-Control": "private, max-age=3600",
            },
        )

    # Fall back to the original uploaded file (image uploads)
    files = [f for f in upload_dir.iterdir() if f.is_file()]
    if not files:
        raise HTTPException(
            status_code=404,
            detail=f"No file found in document directory for '{document_id}'.",
        )

    file_path = files[0]

    # Determine MIME type from extension
    ext_mime = {
        ".pdf":  "application/pdf",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".tiff": "image/tiff",
        ".tif":  "image/tiff",
        ".webp": "image/webp",
    }
    mime = ext_mime.get(file_path.suffix.lower(), "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=mime,
        filename=file_path.name,
        headers={
            "X-Document-ID": document_id,
            "Cache-Control": "private, max-age=3600",
        },
    )


# ===========================================================================
# Health check
# ===========================================================================


@router.get(
    "/health",
    summary="Provider and service health check",
    description=(
        "Check the health of both LLM providers. "
        "Returns 200 if at least one provider is available. "
        "Returns 503 if both are unavailable (pipeline cannot process documents)."
    ),
)
async def health_check() -> JSONResponse:
    """
    Liveness check for the extraction pipeline.

    Runs health checks on both Ollama and Groq providers concurrently.
    Used by load balancers, monitoring, and the startup sequence.

    Returns:
      200 — at least one provider is available (degraded mode possible)
      503 — both providers unavailable (pipeline is down)
    """
    import asyncio

    ollama = OllamaProvider()
    groq   = GroqProvider()

    # Run both health checks concurrently
    ollama_ok, groq_ok = await asyncio.gather(
        ollama.health_check(),
        groq.health_check(),
        return_exceptions=False,
    )

    settings = get_settings()
    both_down = not ollama_ok and not groq_ok

    payload = {
        "status": "degraded" if both_down else "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": {
            "ollama": {
                "healthy": ollama_ok,
                "model": settings.ollama_model,
                "base_url": settings.ollama_base_url,
            },
            "groq": {
                "healthy": groq_ok,
                "model": settings.groq_model,
            },
        },
        "pipeline": {
            "confidence_threshold": settings.confidence_threshold,
            "imap_enabled": settings.imap_enabled,
            "webhooks_enabled": settings.webhooks_enabled,
        },
    }

    return JSONResponse(
        content=payload,
        status_code=503 if both_down else 200,
    )