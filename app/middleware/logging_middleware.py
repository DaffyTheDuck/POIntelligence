"""
app/middleware/logging_middleware.py

Two things in this file:

1. request_id_var — a ContextVar that stores the current request's ID.
   ContextVar is Python's thread-safe / async-safe way to store per-request
   context. Unlike a global variable, each async task gets its own value.
   Any service that imports request_id_var can read the current request ID
   without it being passed explicitly through every function call.

2. RequestLoggingMiddleware — logs every HTTP request with:
   - Method, path, client IP on arrival
   - Status code and duration on completion
   - The request ID in every line
   - X-Request-ID header on every response (useful for debugging from the UI)

Why ContextVar and not threading.local():
   FastAPI is async — multiple requests run on the same thread concurrently.
   threading.local() would give you the wrong value. ContextVar is scoped to
   the asyncio Task, not the thread, so each request gets its own isolated value.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# ── Context variable ──────────────────────────────────────────────────────
# Stores the current request's short ID.
# Default '-' so log lines outside a request context still have a value.
#
# Usage in any service:
#   from app.middleware.logging_middleware import request_id_var
#   rid = request_id_var.get()
#
# You don't need to do this manually — the RequestIDFilter below
# automatically injects it into every log record.

request_id_var: ContextVar[str] = ContextVar('request_id', default='-')


# ── Log filter ────────────────────────────────────────────────────────────

class RequestIDFilter(logging.Filter):
    """
    Injects the current request_id into every log record.

    Once this filter is attached to a handler, every log line from any
    module will include %(request_id)s without any service needing to
    explicitly pass it.

    This is what makes "find all log lines for this request" possible —
    grep for the 8-character request ID and you get the full trace.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


# ── Middleware ────────────────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request/response pair with timing and a correlation ID.

    Per-request flow:
      1. Generate a short request ID (first 8 chars of UUID4)
      2. Set it in request_id_var — now ALL log lines in this request
         will automatically include it via RequestIDFilter
      3. Log the incoming request
      4. Call the next handler (route + services)
      5. Log the response (status code + duration)
      6. Set X-Request-ID header on the response
      7. Reset the ContextVar (cleanup)

    Skips logging for:
      - /health (health checks poll frequently, noisy)
      - /docs, /redoc, /openapi.json (OpenAPI UI assets)
    """

    # Paths to skip logging (too noisy or irrelevant)
    _SKIP_PATHS = frozenset({
        '/api/v1/health',
        '/health',
        '/docs',
        '/redoc',
        '/openapi.json',
        '/favicon.ico',
    })

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip noisy paths
        if request.url.path in self._SKIP_PATHS:
            return await call_next(request)

        # Generate and store request ID
        request_id = uuid.uuid4().hex[:8]
        token = request_id_var.set(request_id)

        # Capture client IP (handles reverse-proxy X-Forwarded-For)
        client_ip = (
            request.headers.get('X-Forwarded-For', '').split(',')[0].strip()
            or (request.client.host if request.client else 'unknown')
        )

        start_ns = time.perf_counter_ns()

        # Log incoming request
        logger.info(
            '→ %s %s  client=%s  rid=%s',
            request.method,
            request.url.path,
            client_ip,
            request_id,
        )

        # Process the request
        status_code = 500  # default if exception is raised
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response

        except Exception as exc:
            logger.error(
                '✗ %s %s  rid=%s  error=%s',
                request.method,
                request.url.path,
                request_id,
                exc,
                exc_info=True,
            )
            raise

        finally:
            duration_ms = (time.perf_counter_ns() - start_ns) // 1_000_000

            # Choose log level by status code
            if status_code >= 500:
                log = logger.error
            elif status_code >= 400:
                log = logger.warning
            else:
                log = logger.info

            log(
                '← %s %s  status=%d  duration=%dms  rid=%s',
                request.method,
                request.url.path,
                status_code,
                duration_ms,
                request_id,
            )

            # Add correlation ID to response headers so the frontend
            # can include it in bug reports
            try:
                response.headers['X-Request-ID'] = request_id
            except Exception:
                pass  # Response may already be sent

            # Clean up ContextVar
            request_id_var.reset(token)