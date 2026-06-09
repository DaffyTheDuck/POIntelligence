"""
app/config.py

Single source of truth for all runtime configuration.
Loaded once at startup via get_settings() and cached — never re-read from disk mid-request.

Usage (in any service or route):
    from app.config import get_settings
    settings = get_settings()

Environment variables are loaded from a .env file in the project root.
See .env.example for all required and optional values.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.po_models import CONFIDENCE_THRESHOLD


class Settings(BaseSettings):
    """
    All configuration for the PO Intelligence pipeline.

    Pydantic-settings reads values in this priority order (highest to lowest):
      1. Environment variables (set in shell or CI)
      2. .env file in the project root
      3. Default values defined here

    Type coercion is automatic — "true" becomes True, "0.75" becomes 0.75.
    Missing required fields (no default, no env var) raise a ValidationError at startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # REDIS_URL and redis_url are the same
        extra="ignore",         # Silently ignore env vars not declared here
    )

    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------

    app_name: str = Field(
        default="PO Intelligence",
        description="Human-readable app name, used in API docs and log lines"
    )
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(
        default=False,
        description="Enables FastAPI debug mode and verbose logging. Never True in prod."
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level: DEBUG, INFO, WARNING, ERROR"
    )

    # -----------------------------------------------------------------------
    # Ollama — local model on the Linux GPU machine
    # -----------------------------------------------------------------------

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description=(
            "Base URL of the Ollama server. "
            "On the Windows dev machine, set this to the Linux machine's LAN IP, e.g. "
            "http://192.168.1.x:11434. On the Linux machine itself, localhost is correct."
        )
    )
    ollama_model: str = Field(
        default="llava-phi3",
        description="Ollama model tag to use for extraction. Must be pulled on the Linux machine."
    )
    ollama_timeout_seconds: int = Field(
        default=1800,
        ge=10,
        le=7200,
        description=(
            "Max seconds to wait for an Ollama response before treating it as failed. "
            "Set high (1800+) when testing local vision models on limited VRAM — "
            "a cold llava-phi3 inference on a GTX 1650 can take several minutes."
        )
    )
    ollama_max_retries: int = Field(
        default=0,
        ge=0,
        le=5,
        description=(
            "Number of times to retry a failed Ollama request before falling back to Groq. "
            "Set to 0 when testing with long timeouts — retries multiply the wait time."
        )
    )

    # -----------------------------------------------------------------------
    # Fallback API provider — confidence-based (architecture decision #3)
    # Active provider: Groq (free, fast, vision support)
    # Switch provider by changing the import in router_service.py
    # -----------------------------------------------------------------------

    # Groq — active fallback provider
    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key. Get free key at console.groq.com"
    )
    groq_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="Groq model. Must support vision. See console.groq.com/docs/models"
    )

    # Claude — kept optional in case you want to switch back
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key. Optional — only needed if switching back to ClaudeProvider."
    )
    claude_model: str = Field(
        default="claude-opus-4-5",
        description="Claude model name. Only used if ClaudeProvider is active in router_service.py."
    )

    # Shared fallback settings — apply to whichever fallback provider is active
    claude_timeout_seconds: int = Field(
        default=90,
        ge=10,
        le=300,
        description="Max seconds to wait for the fallback API provider (Groq)."
    )
    claude_max_tokens: int = Field(
        default=4096,
        description="Max tokens in fallback provider response."
    )

    # -----------------------------------------------------------------------
    # Confidence routing (architecture decision #3)
    # -----------------------------------------------------------------------

    confidence_threshold: float = Field(
        default=CONFIDENCE_THRESHOLD,  # 0.75, imported from po_models
        ge=0.0,
        le=1.0,
        description=(
            "Fields with confidence below this score trigger Groq fallback. "
            "Set to 0.0 to always use local model only. "
            "Set to 1.0 to always escalate everything to Groq (useful for demos)."
        )
    )
    disagreement_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "If |local_confidence - groq_confidence| exceeds this, flag for human review. "
            "Architecture decision #4."
        )
    )

    # -----------------------------------------------------------------------
    # Redis — Celery broker + result backend (architecture decision #6)
    # -----------------------------------------------------------------------

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description=(
            "Redis connection URL. Used as both Celery broker and result backend. "
            "On the Windows dev machine, point this to the Linux machine's Redis instance."
        )
    )
    redis_job_ttl_seconds: int = Field(
        default=86400,  # 24 hours
        description="How long to keep completed job records in Redis before expiry."
    )

    # -----------------------------------------------------------------------
    # Celery (architecture decision #6)
    # -----------------------------------------------------------------------

    celery_worker_concurrency: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of concurrent Celery tasks. "
            "Keep at 1 — the GTX 1650 has 4GB VRAM. Running two llava-phi3 "
            "inferences simultaneously will OOM. Queue protects the GPU."
        )
    )
    celery_task_timeout_seconds: int = Field(
        default=600,
        description=(
            "Hard timeout for a single Celery task. Kills zombie tasks. "
            "Set to 600 — qwen2.5vl:3b on GTX 1650 takes ~300s; "
            "600 gives 2x headroom without letting tasks hang forever."
        )
    )

    # -----------------------------------------------------------------------
    # surya OCR (architecture decision #7)
    # -----------------------------------------------------------------------

    surya_device: str = Field(
        default="cuda",
        description=(
            "Device for surya inference: 'cuda' on the Linux GPU machine, "
            "'cpu' on the Windows dev machine. "
            "Set to 'cpu' automatically if CUDA is unavailable — see validator below."
        )
    )
    surya_batch_size: int = Field(
        default=1,
        ge=1,
        description="Number of pages to process per surya batch. 1 is safe for 4GB VRAM."
    )
    surya_detector_thresh: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Detection confidence threshold for surya's DetectionPredictor. "
            "Lower values (0.1) detect more regions on scanned/low-contrast documents, "
            "reducing fallback to horizontal strip OCR. "
            "Higher values (0.3+) are more conservative. "
            "Set via SURYA_DETECTOR_THRESH in .env."
        )
    )
    ocr_fuzzy_match_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fuzzy match ratio to accept a bounding box as a match (layer 2 OCR). "
            "Architecture decision #8: below this, falls through to LLM spatial hint."
        )
    )

    # -----------------------------------------------------------------------
    # File upload
    # -----------------------------------------------------------------------

    upload_dir: str = Field(
        default="./uploads",
        description="Local directory where uploaded documents are stored before processing."
    )
    max_upload_size_mb: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum upload file size in megabytes."
    )
    allowed_mime_types: List[str] = Field(
        default=[
            "application/pdf",
            "image/jpeg",
            "image/png",
            "image/tiff",
            "image/webp",
        ],
        description="MIME types accepted at the upload endpoint."
    )

    @property
    def max_upload_size_bytes(self) -> int:
        """Convenience property — routes use bytes, config stores MB."""
        return self.max_upload_size_mb * 1024 * 1024

    # -----------------------------------------------------------------------
    # IMAP email ingestion (architecture decision #1 — async path)
    # -----------------------------------------------------------------------

    imap_host: str | None = Field(
        default=None,
        description="IMAP server hostname, e.g. imap.gmail.com"
    )
    imap_port: int = Field(
        default=993,
        description="IMAP port. 993 = IMAPS (TLS). 143 = plain IMAP."
    )
    imap_username: str | None = Field(
        default=None,
        description="Email address / IMAP login username."
    )
    imap_password: str | None = Field(
        default=None,
        description="IMAP password or app-specific password (for Gmail, use an App Password)."
    )
    imap_mailbox: str = Field(
        default="INBOX",
        description="Which mailbox to monitor for incoming PO documents."
    )
    imap_poll_interval_seconds: int = Field(
        default=60,
        ge=10,
        description="How often the email service polls for new messages."
    )
    imap_attachment_subject_filter: str | None = Field(
        default=None,
        description=(
            "Optional subject line substring filter. "
            "e.g. 'Purchase Order' — only emails matching this are processed. "
            "None means process all emails with PDF/image attachments."
        )
    )

    # -----------------------------------------------------------------------
    # Export (architecture decision #9 — GDPR: generated locally)
    # -----------------------------------------------------------------------

    export_dir: str = Field(
        default="./exports",
        description=(
            "Directory where export files are written. "
            "Files are generated locally and never uploaded to a third party — GDPR compliance."
        )
    )
    webhook_url: str | None = Field(
        default=None,
        description=(
            "ERP or downstream system webhook endpoint. "
            "None disables webhooks — export-only mode. "
            "Architecture decision #9: loose coupling, zero ERP lock-in."
        )
    )
    webhook_timeout_seconds: int = Field(
        default=10,
        description="Max seconds to wait for the webhook endpoint to respond."
    )
    webhook_secret: str | None = Field(
        default=None,
        description=(
            "Optional HMAC secret for signing webhook payloads. "
            "If set, a X-Signature header is included on every webhook POST."
        )
    )

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

    @field_validator("log_level", mode="before")
    @classmethod
    def normalise_log_level(cls, v: str) -> str:
        """Accept any case — 'debug', 'DEBUG', 'Debug' all work."""
        return v.upper()

    @field_validator("surya_device", mode="before")
    @classmethod
    def validate_surya_device(cls, v: str) -> str:
        allowed = {"cuda", "cpu", "mps"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"surya_device must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def warn_if_imap_partial(self) -> "Settings":
        """
        IMAP config is optional as a whole, but if you set any IMAP field
        you must set all required ones. Catches typos in .env early.
        """
        imap_fields = [self.imap_host, self.imap_username, self.imap_password]
        any_set = any(f is not None for f in imap_fields)
        all_set = all(f is not None for f in imap_fields)
        if any_set and not all_set:
            raise ValueError(
                "Partial IMAP config detected. "
                "Set all of IMAP_HOST, IMAP_USERNAME, IMAP_PASSWORD — or none of them."
            )
        return self

    @model_validator(mode="after")
    def warn_if_debug_in_prod(self) -> "Settings":
        """
        Not a hard block, but catches the common mistake of committing debug=True.
        In a real deployment you'd check NODE_ENV or ENVIRONMENT here.
        """
        if self.debug:
            import warnings
            warnings.warn(
                "DEBUG=true is set. Never use this in production.",
                stacklevel=2
            )
        return self

    # -----------------------------------------------------------------------
    # Derived properties — computed from raw config, used by services
    # -----------------------------------------------------------------------

    @property
    def imap_enabled(self) -> bool:
        """True only when all required IMAP fields are present."""
        return all([self.imap_host, self.imap_username, self.imap_password])

    @property
    def webhooks_enabled(self) -> bool:
        """True only when a non-empty webhook URL is configured."""
        return bool(self.webhook_url and self.webhook_url.strip())

    @property
    def celery_broker_url(self) -> str:
        """Celery expects the broker URL directly — same Redis instance."""
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        """Result backend is also Redis — job results stored alongside tasks."""
        return self.redis_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance, created once and cached forever.

    Using lru_cache means:
      - .env is read exactly once at first call, not on every request
      - All services share the same object (no drift between instances)
      - In tests, you can override with: get_settings.cache_clear() then monkeypatch

    Never instantiate Settings() directly in service code.
    Always call get_settings() so tests can override cleanly.
    """
    return Settings()