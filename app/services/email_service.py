"""
app/services/email_service.py

Async email ingestion path (architecture decision #1).

Architecture decision #1 in practice:
  Upload path:  route → document_service.process_upload() → ExtractionResult (sync)
  Email path:   email_service polls IMAP → queues Celery task → client polls job status

The extraction pipeline (ocr → route → extract → validate) is identical.
The difference is purely in how the document enters the system and how
the result is returned to the caller.

Three components in this file:
  1. JobStore     — Redis-backed storage for JobRecord lifecycle tracking
  2. EmailService — IMAP polling loop + attachment dispatch
  3. Celery task  — runs the extraction pipeline in a worker thread

Celery / asyncio note:
  Celery workers are synchronous threads. The extraction pipeline is async.
  The task calls asyncio.run() to create a fresh event loop per task.
  This is safe because CELERY_WORKER_CONCURRENCY=1 (architecture decision #6) —
  only one task runs at a time, so there is no event loop collision.
  surya model loading uses threading.Lock (not asyncio.Lock) for this reason.
"""

from __future__ import annotations

import asyncio
import email as email_lib
import imaplib
import logging
import threading
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

import redis as redis_lib

from app.config import get_settings
from app.models.po_models import (
    JobRecord,
    JobStatus,
    JobStatusResponse,
)
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

from celery import Celery  # type: ignore[import]

_settings = get_settings()
celery_app = Celery(
    "po_intelligence",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=_settings.celery_worker_concurrency,
    task_soft_time_limit=_settings.celery_task_timeout_seconds,
    task_time_limit=_settings.celery_task_timeout_seconds + 30,
    # Retry failed tasks once after 60s — handles transient Ollama/Claude failures
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)


# ---------------------------------------------------------------------------
# Job store — Redis-backed JobRecord persistence
# ---------------------------------------------------------------------------


class JobStore:
    """
    Stores and retrieves JobRecord objects in Redis.

    Key pattern: "job:{job_id}"
    TTL: redis_job_ttl_seconds (default 24 hours) from config.

    Falls back to an in-memory dict if Redis is unavailable — useful for
    development on the Windows machine before Redis is running.

    Thread-safe: Redis operations are thread-safe by default.
    Safe for both async routes (via run_in_executor) and sync Celery tasks.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis: Optional[redis_lib.Redis] = None
        self._fallback: dict = {}  # In-memory fallback
        self._lock = threading.Lock()
        self._connect()

    def _connect(self) -> None:
        try:
            self._redis = redis_lib.from_url(
                self._settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._redis.ping()
            logger.info("JobStore connected to Redis at %s", self._settings.redis_url)
        except Exception as e:
            logger.warning(
                "Redis unavailable (%s). JobStore using in-memory fallback. "
                "Start Redis or set REDIS_URL correctly for persistent job tracking.",
                e,
            )
            self._redis = None

    def save(self, job: JobRecord) -> None:
        """Persist a new JobRecord."""
        ttl = self._settings.redis_job_ttl_seconds
        serialised = job.model_dump_json()
        if self._redis:
            try:
                self._redis.setex(f"job:{job.job_id}", ttl, serialised)
                return
            except Exception as e:
                logger.warning("Redis write failed: %s. Using in-memory fallback.", e)
        with self._lock:
            self._fallback[job.job_id] = serialised

    def get(self, job_id: str) -> Optional[JobRecord]:
        """Retrieve a JobRecord by job_id."""
        if self._redis:
            try:
                data = self._redis.get(f"job:{job_id}")
                if data:
                    return JobRecord.model_validate_json(data)
                return None
            except Exception as e:
                logger.warning("Redis read failed: %s. Checking in-memory fallback.", e)
        with self._lock:
            data = self._fallback.get(job_id)
            return JobRecord.model_validate_json(data) if data else None

    def update(self, job: JobRecord) -> None:
        """Update an existing JobRecord (status change, result added, etc.)."""
        self.save(job)  # setex overwrites existing key

    def to_status_response(self, job: JobRecord) -> JobStatusResponse:
        """Convert a JobRecord to the API response shape."""
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            is_terminal=job.is_terminal,
            result=job.result,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


# Module-level singleton — shared by EmailService, Celery task, and routes
job_store = JobStore()


# ---------------------------------------------------------------------------
# Celery task — runs the extraction pipeline in a worker thread
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="po_intelligence.extract_document",
    max_retries=1,
    default_retry_delay=60,
)
def extract_document_task(
    self,
    job_id: str,
    document_id: str,
    file_path: str,
    mime_type: str,
) -> None:
    """
    Celery task: run the extraction pipeline for an email-ingested document.

    Called by EmailService._queue_attachment() after the attachment is saved.
    Updates JobRecord status at each lifecycle stage.

    asyncio.run() creates a fresh event loop for this thread.
    Safe with CELERY_WORKER_CONCURRENCY=1 (one task at a time = no loop collision).

    On failure: updates job to FAILED and retries once after 60s.
    If the retry also fails: job stays FAILED, client is informed via polling.
    """
    settings = get_settings()

    # ── Mark as PROCESSING ──────────────────────────────────────────────
    job = job_store.get(job_id)
    if job is None:
        logger.error("Task started for unknown job_id=%s. Aborting.", job_id)
        return

    job = job.model_copy(update={
        "status": JobStatus.PROCESSING,
        "updated_at": datetime.now(timezone.utc),
    })
    job_store.update(job)
    logger.info("Task started: job_id=%s, document_id=%s", job_id, document_id)

    # ── Run extraction pipeline ──────────────────────────────────────────
    try:
        # ExtractionService is async — run in a fresh event loop
        from app.services.extraction_service import ExtractionService
        extraction_service = ExtractionService()

        result = asyncio.run(
            extraction_service.extract(
                file_path=file_path,
                document_id=document_id,
                mime_type=mime_type,
            )
        )

        # Store the result in DocumentService's result store
        from app.services.document_service import _store as doc_store
        doc_store.save(result)

        # ── Mark as COMPLETE ────────────────────────────────────────────
        job = job.model_copy(update={
            "status": JobStatus.COMPLETE,
            "document_id": document_id,
            "result": result,
            "completed_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        })
        job_store.update(job)

        logger.info(
            "Task complete: job_id=%s, confidence=%.2f, flagged=%d",
            job_id, result.overall_confidence, len(result.fields_flagged_for_review),
        )

    except Exception as exc:
        logger.error(
            "Task failed: job_id=%s, error=%s. Retrying if attempts remain.",
            job_id, exc, exc_info=True,
        )

        # ── Mark as FAILED ──────────────────────────────────────────────
        job = job.model_copy(update={
            "status": JobStatus.FAILED,
            "error_message": str(exc),
            "updated_at": datetime.now(timezone.utc),
        })
        job_store.update(job)

        # Retry once after 60s (for transient failures like Ollama cold start)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            logger.error(
                "Task max retries exceeded: job_id=%s. Job permanently failed.", job_id
            )


# ---------------------------------------------------------------------------
# Email service — IMAP polling loop
# ---------------------------------------------------------------------------


class EmailService:
    """
    Polls an IMAP mailbox for new emails with PDF/image attachments
    and dispatches each attachment as a Celery extraction task.

    Lifecycle:
      start_polling() → runs forever as an asyncio background task
      stop()          → signals the loop to exit cleanly

    IMAP connection is opened and closed on every poll cycle — no persistent
    connection. Slower than IMAP IDLE but simpler: no keepalive, no timeout,
    no reconnect logic on idle disconnect.
    """

    def __init__(
        self,
        document_service: Optional[DocumentService] = None,
    ) -> None:
        self._settings = get_settings()
        self._doc_service = document_service or DocumentService()
        self._running = False
        self._upload_dir = Path(self._settings.upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    async def start_polling(self) -> None:
        """
        Async polling loop — run as a background task from main.py lifespan.

        Exits cleanly when stop() is called or when the app shuts down.
        IMAP is disabled if IMAP_HOST is not configured — logs and returns.
        """
        if not self._settings.imap_enabled:
            logger.info(
                "IMAP not configured — email ingestion disabled. "
                "Set IMAP_HOST, IMAP_USERNAME, IMAP_PASSWORD to enable."
            )
            return

        self._running = True
        logger.info(
            "Email polling started: host=%s, mailbox=%s, interval=%ds",
            self._settings.imap_host,
            self._settings.imap_mailbox,
            self._settings.imap_poll_interval_seconds,
        )

        while self._running:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, self._poll_sync
                )
            except Exception as e:
                # Log but don't crash — next poll will retry
                logger.error(
                    "IMAP poll cycle failed: %s. "
                    "Will retry in %ds.",
                    e, self._settings.imap_poll_interval_seconds,
                )

            await asyncio.sleep(self._settings.imap_poll_interval_seconds)

    def stop(self) -> None:
        """Signal the polling loop to exit after the current cycle."""
        self._running = False
        logger.info("Email polling stop requested.")

    # ------------------------------------------------------------------
    # Sync IMAP interaction (runs in thread pool)
    # ------------------------------------------------------------------

    def _poll_sync(self) -> None:
        """
        One complete poll cycle:
          1. Connect and authenticate
          2. Fetch unseen messages (with optional subject filter)
          3. Extract PDF/image attachments
          4. Save attachment to disk and queue Celery task
          5. Mark message as seen
          6. Disconnect

        Each step is wrapped so a failure on one message doesn't abort the cycle.
        """
        imap = self._connect_imap()
        if imap is None:
            return

        try:
            messages = self._fetch_unseen_message_ids(imap)
            if not messages:
                logger.debug("No unseen messages found.")
                return

            logger.info("Found %d unseen message(s) to process.", len(messages))

            for msg_id in messages:
                try:
                    self._process_message(imap, msg_id)
                except Exception as e:
                    logger.error(
                        "Failed to process message id=%s: %s. Skipping.",
                        msg_id, e
                    )
        finally:
            try:
                imap.logout()
            except Exception:
                pass  # Best-effort logout

    def _connect_imap(self) -> Optional[imaplib.IMAP4_SSL]:
        """
        Open an authenticated IMAP4_SSL connection.
        Returns None on failure — caller skips the poll cycle.
        """
        try:
            imap = imaplib.IMAP4_SSL(
                self._settings.imap_host,
                self._settings.imap_port,
            )
            imap.login(
                self._settings.imap_username,
                self._settings.imap_password,
            )
            imap.select(self._settings.imap_mailbox)
            logger.debug(
                "IMAP connected: host=%s, mailbox=%s",
                self._settings.imap_host,
                self._settings.imap_mailbox,
            )
            return imap
        except imaplib.IMAP4.error as e:
            logger.error(
                "IMAP authentication failed: %s. "
                "Check IMAP_USERNAME and IMAP_PASSWORD (Gmail needs an App Password).",
                e,
            )
            return None
        except OSError as e:
            logger.error(
                "Cannot connect to IMAP server %s:%d — %s. "
                "Check IMAP_HOST and network connectivity.",
                self._settings.imap_host, self._settings.imap_port, e,
            )
            return None

    def _fetch_unseen_message_ids(
        self, imap: imaplib.IMAP4_SSL
    ) -> List[bytes]:
        """
        Return a list of IMAP message IDs for unseen messages.
        Applies subject filter if IMAP_ATTACHMENT_SUBJECT_FILTER is configured.
        """
        # Search for UNSEEN messages (not yet marked \\Seen)
        _, data = imap.search(None, "UNSEEN")
        if not data or not data[0]:
            return []

        msg_ids = data[0].split()

        # Subject filter — fetch headers only for matching subjects
        subject_filter = self._settings.imap_attachment_subject_filter
        if not subject_filter:
            return msg_ids

        matching: List[bytes] = []
        for msg_id in msg_ids:
            _, header_data = imap.fetch(msg_id, "(BODY[HEADER.FIELDS (SUBJECT)])")
            if not header_data or not header_data[0]:
                continue
            raw_header = header_data[0][1]
            msg = email_lib.message_from_bytes(raw_header)
            subject = msg.get("subject", "")
            if subject_filter.lower() in subject.lower():
                matching.append(msg_id)
            else:
                logger.debug(
                    "Skipping message id=%s: subject '%s' does not match filter '%s'",
                    msg_id, subject, subject_filter,
                )

        return matching

    def _process_message(
        self, imap: imaplib.IMAP4_SSL, msg_id: bytes
    ) -> None:
        """
        Fetch, parse, and process one email message.
        Marks it as seen regardless of whether attachments were found.
        """
        _, msg_data = imap.fetch(msg_id, "(RFC822)")
        if not msg_data or not msg_data[0]:
            logger.warning("Empty message data for id=%s", msg_id)
            return

        raw_email = msg_data[0][1]
        msg = email_lib.message_from_bytes(raw_email)

        sender = msg.get("from", "unknown")
        subject = msg.get("subject", "(no subject)")
        logger.info(
            "Processing email: from='%s', subject='%s'", sender, subject
        )

        attachments = self._extract_attachments(msg)
        if not attachments:
            logger.info(
                "No valid attachments in email from '%s' (subject: '%s'). Skipping.",
                sender, subject,
            )
        else:
            for filename, file_bytes, mime_type in attachments:
                try:
                    self._queue_attachment(
                        filename=filename,
                        file_bytes=file_bytes,
                        mime_type=mime_type,
                        source_email=sender,
                        source_subject=subject,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to queue attachment '%s' from '%s': %s",
                        filename, sender, e,
                    )

        # Mark as seen — even if no valid attachments, don't re-process
        imap.store(msg_id, "+FLAGS", "\\Seen")

    def _extract_attachments(
        self, msg: Message
    ) -> List[Tuple[str, bytes, str]]:
        """
        Extract PDF and image attachments from an email message.

        Returns list of (filename, file_bytes, mime_type) tuples.
        Skips inline images, oversized files, and unsupported types.
        """
        attachments: List[Tuple[str, bytes, str]] = []
        allowed_types = set(self._settings.allowed_mime_types)
        max_bytes = self._settings.max_upload_size_bytes

        for part in msg.walk():
            # Skip multipart containers
            content_maintype = part.get_content_maintype()
            if content_maintype == "multipart":
                continue

            # Only process attachments (Content-Disposition: attachment)
            disposition = part.get("Content-Disposition", "")
            if "attachment" not in disposition.lower():
                continue

            filename = part.get_filename()
            if not filename:
                continue

            mime_type = part.get_content_type().lower()
            if mime_type not in allowed_types:
                logger.debug(
                    "Skipping attachment '%s': unsupported type '%s'",
                    filename, mime_type,
                )
                continue

            file_bytes = part.get_payload(decode=True)
            if not file_bytes:
                logger.debug("Skipping attachment '%s': empty payload", filename)
                continue

            if len(file_bytes) > max_bytes:
                logger.warning(
                    "Skipping attachment '%s': size %dMB exceeds %dMB limit.",
                    filename,
                    len(file_bytes) // (1024 * 1024),
                    self._settings.max_upload_size_mb,
                )
                continue

            attachments.append((filename, file_bytes, mime_type))
            logger.debug(
                "Found attachment: '%s', type=%s, size=%d bytes",
                filename, mime_type, len(file_bytes),
            )

        return attachments

    def _queue_attachment(
        self,
        filename: str,
        file_bytes: bytes,
        mime_type: str,
        source_email: str,
        source_subject: str,
    ) -> str:
        """
        Save attachment to disk, create a JobRecord, and dispatch a Celery task.

        Returns the job_id so it can be used in any logging or notification.
        The client polls GET /api/v1/jobs/{job_id} for status updates.
        """
        document_id = str(uuid4())
        job_id = str(uuid4())

        # Save the file to disk (same structure as upload path)
        safe_filename = Path(filename).name
        doc_dir = self._upload_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        file_path = doc_dir / safe_filename
        file_path.write_bytes(file_bytes)

        logger.info(
            "Attachment saved: job_id=%s, document_id=%s, path=%s",
            job_id, document_id, file_path,
        )

        # Create and persist a JobRecord immediately
        # Client can start polling right away — before the task even starts
        job = JobRecord(
            job_id=job_id,
            document_id=document_id,
            status=JobStatus.PENDING,
            source_email=source_email,
            source_subject=source_subject,
            filename=safe_filename,
        )
        job_store.save(job)

        # Dispatch Celery task
        extract_document_task.apply_async(
            kwargs={
                "job_id": job_id,
                "document_id": document_id,
                "file_path": str(file_path),
                "mime_type": mime_type,
            },
            task_id=job_id,  # Use job_id as Celery task ID for correlation
        )

        logger.info(
            "Celery task dispatched: job_id=%s, filename='%s', from='%s'",
            job_id, filename, source_email,
        )

        return job_id