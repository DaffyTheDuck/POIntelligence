"""
app/middleware/task_logging.py

Logging context for Celery tasks — mirrors what RequestLoggingMiddleware
does for HTTP requests, but for async worker tasks.

Usage in a Celery task:
    from app.middleware.task_logging import task_log_context

    @celery_app.task(bind=True)
    def my_task(self, job_id, document_id):
        with task_log_context(job_id=job_id, document_id=document_id):
            logger.info("Starting task")   # → includes job_id in output
            do_work()
            logger.info("Task done")       # → includes job_id in output

The context manager sets request_id_var to the job_id so the same
RequestIDFilter that works for HTTP requests also works for tasks.
All log lines inside the context block will show the job_id in the
request_id column of the log format.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from app.middleware.logging_middleware import request_id_var

logger = logging.getLogger(__name__)


@contextmanager
def task_log_context(
    job_id: str,
    document_id: str | None = None,
) -> Generator[None, None, None]:
    """
    Context manager that injects job_id into all log lines.

    Sets request_id_var to a short tag like "job:a1b2c3d4" so log output
    looks like:
      2024-01-15 14:23:01 | INFO | job:a1b2 | app.services.extraction_service | OCR complete
      2024-01-15 14:23:03 | INFO | job:a1b2 | app.providers.ollama_provider | Inference complete

    The job_id prefix makes it immediately obvious in mixed logs which
    lines came from HTTP requests and which came from Celery tasks.
    """
    # Use first 8 chars of job_id — same length as HTTP request IDs
    tag = f"job:{job_id[:8]}"
    token = request_id_var.set(tag)

    extra_info = f"job_id={job_id}"
    if document_id:
        extra_info += f" document_id={document_id}"

    logger.info("Task context started: %s", extra_info)
    try:
        yield
    except Exception as exc:
        logger.error(
            "Task context failed: %s  error=%s",
            extra_info, exc,
            exc_info=True,
        )
        raise
    finally:
        logger.info("Task context ended: %s", extra_info)
        request_id_var.reset(token)