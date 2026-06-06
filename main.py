"""
main.py

FastAPI application entrypoint.

Run on Linux:
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

Start Celery worker separately:
  celery -A app.services.email_service.celery_app worker --loglevel=info --concurrency=1
"""

from __future__ import annotations

import asyncio
import logging
import logging.config
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.middleware.logging_middleware import (
    RequestLoggingMiddleware,
    RequestIDFilter,
)

# ---------------------------------------------------------------------------
# Logging — configured before anything else
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    """
    Configure structured logging with request ID injection.

    Format includes:
      timestamp | level | request_id | logger_name | message

    The request_id field is populated by RequestIDFilter from the
    ContextVar set by RequestLoggingMiddleware. Outside a request
    context (startup, Celery tasks) it shows '-'.

    Example output:
      2024-01-15 14:23:01 | INFO     | a1b2c3d4 | app.services.router_service | Primary provider selected: phi3.5-vision
      2024-01-15 14:23:04 | INFO     | a1b2c3d4 | app.api.routes | <- POST /api/v1/documents/upload status=200 duration=3241ms rid=a1b2c3d4
    """
    request_id_filter = RequestIDFilter()

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'filters': {
            'request_id': {
                '()': lambda: request_id_filter,
            },
        },
        'formatters': {
            'standard': {
                'format': (
                    '%(asctime)s | %(levelname)-8s | %(request_id)-8s | '
                    '%(name)s | %(message)s'
                ),
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'filters': ['request_id'],
                'stream': 'ext://sys.stdout',
            },
        },
        'root': {
            'level': level,
            'handlers': ['console'],
        },
        # Quiet noisy third-party loggers
        'loggers': {
            'httpx':          {'level': 'WARNING'},
            'httpcore':       {'level': 'WARNING'},
            'celery':         {'level': 'INFO'},
            'celery.task':    {'level': 'INFO'},
            'PIL':            {'level': 'WARNING'},
            'surya':          {'level': 'INFO'},
            'uvicorn':        {'level': 'INFO'},
            'uvicorn.access': {'level': 'WARNING'},  # replaced by our middleware
            'uvicorn.error':  {'level': 'INFO'},
        },
    })


_settings = get_settings()
_configure_logging(_settings.log_level)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: warm surya, start IMAP. Shutdown: stop polling cleanly."""

    logger.info('=' * 60)
    logger.info('PO Intelligence %s starting', _settings.app_version)
    logger.info('  Ollama:     %s  model=%s', _settings.ollama_base_url, _settings.ollama_model)
    logger.info('  Claude:     model=%s', _settings.claude_model)
    logger.info('  Redis:      %s', _settings.redis_url)
    logger.info('  Threshold:  %.2f', _settings.confidence_threshold)
    logger.info('  IMAP:       %s', 'enabled' if _settings.imap_enabled else 'disabled')
    logger.info('  Webhooks:   %s', 'enabled' if _settings.webhooks_enabled else 'disabled')
    logger.info('  Debug:      %s', _settings.debug)
    logger.info('  Log level:  %s', _settings.log_level)
    logger.info('=' * 60)

    # Warm up surya models
    from app.services.ocr_service import OCRService
    ocr_service = OCRService()
    try:
        await ocr_service.warm_up()
    except Exception as e:
        logger.warning(
            'surya warm-up failed: %s — OCR will use embedded PDF text only.', e
        )

    # Start IMAP polling loop
    from app.services.email_service import EmailService
    email_service = EmailService()
    polling_task: asyncio.Task | None = None

    if _settings.imap_enabled:
        polling_task = asyncio.create_task(
            email_service.start_polling(),
            name='imap_polling_loop',
        )
        logger.info(
            'IMAP polling started: host=%s interval=%ds',
            _settings.imap_host,
            _settings.imap_poll_interval_seconds,
        )
    else:
        logger.info('IMAP disabled — set IMAP_HOST, IMAP_USERNAME, IMAP_PASSWORD to enable.')

    app.state.email_service = email_service
    app.state.ocr_service = ocr_service
    logger.info('Startup complete. Accepting requests.')

    yield  # Application runs

    # Shutdown
    logger.info('Shutting down...')
    email_service.stop()
    if polling_task and not polling_task.done():
        polling_task.cancel()
        try:
            await polling_task
        except asyncio.CancelledError:
            logger.info('IMAP polling task cancelled.')
    logger.info('Shutdown complete.')


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title='PO Intelligence',
    description=(
        'Purchase Order extraction pipeline. '
        'Local-first (phi3.5-vision) with Claude confidence-based fallback.'
    ),
    version=_settings.app_version,
    lifespan=lifespan,
    docs_url='/docs',
    redoc_url='/redoc',
    openapi_url='/openapi.json',
)


# ---------------------------------------------------------------------------
# Middleware — order matters: added last = runs first
# ---------------------------------------------------------------------------

# 1. Request logging + correlation ID (innermost — runs closest to the route)
app.add_middleware(RequestLoggingMiddleware)

# 2. CORS (outermost — handles preflight before logging)
import os
if _settings.debug:
    _cors_origins = ['*']
else:
    _env = os.getenv('CORS_ORIGINS', '')
    _cors_origins = (
        [o.strip() for o in _env.split(',') if o.strip()]
        if _env else [
            'http://localhost:3000',
            'http://localhost:5173',
            'http://127.0.0.1:3000',
            'http://127.0.0.1:5173',
        ]
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allow_headers=['*'],
    expose_headers=['X-Request-ID'],  # expose so frontend can read it
)


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        'Unhandled exception on %s %s: %s',
        request.method, request.url.path, exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            'detail': 'Internal server error.',
            'path': str(request.url.path),
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(router, prefix='/api/v1', tags=['PO Intelligence'])


@app.get('/', include_in_schema=False)
async def root() -> dict:
    return {
        'service': 'PO Intelligence',
        'version': _settings.app_version,
        'docs': '/docs',
        'health': '/api/v1/health',
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=8000,
        reload=_settings.debug,
        workers=1,
        log_level=_settings.log_level.lower(),
    )