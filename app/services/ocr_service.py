"""
app/services/ocr_service.py

Two responsibilities, called at different pipeline stages:

  Stage 1 — Before extraction:
    process_document(file_path, mime_type)
      → Converts PDF pages to images (if needed)
      → Tries embedded text extraction first (fast path for digital PDFs)
      → Falls back to surya OCR for scanned documents
      → Returns OCRDocument with full text + per-line bounding boxes

  Stage 2 — After extraction:
    attach_bounding_boxes(field_extractions, ocr_doc)
      → For each extracted field value, find its location in the document
      → Three-layer matching (architecture decision #8):
          Layer 1: Exact string match
          Layer 2: Fuzzy normalised match (rapidfuzz)
          Layer 3: LLM spatial hint via Ollama (last resort)
      → Returns field_extractions dict with BoundingBox populated

surya models are loaded once as a lazy singleton.
PDF conversion uses PyMuPDF (fitz) — fast and handles encrypted/malformed PDFs better
than pdf2image/poppler.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx # pyright: ignore[reportMissingImports]

from app.config import get_settings
from app.models.po_models import (
    BoundingBox,
    FieldExtraction,
    OCRMatchMethod,
    ReviewReason,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal data structures — not Pydantic, these never leave this layer
# ---------------------------------------------------------------------------


@dataclass
class TextLine:
    """
    A single detected text region from surya.

    Wraps surya's internal TextLine into our own type so the rest of the
    codebase never imports from surya directly. If we swap OCR backends,
    only this file changes.
    """
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int                    # 1-indexed
    ocr_confidence: float = 1.0  # surya's own confidence for this line

    @property
    def normalised_text(self) -> str:
        """Lowercase, stripped, punctuation-collapsed — used for fuzzy matching."""
        return _normalise(self.text)

    def to_bounding_box(self, match_method: OCRMatchMethod) -> BoundingBox:
        return BoundingBox(
            x0=self.x0, y0=self.y0,
            x1=self.x1, y1=self.y1,
            page=self.page,
            ocr_match_method=match_method,
        )


@dataclass
class OCRPage:
    """One page of OCR output."""
    page_number: int        # 1-indexed
    image_bytes: bytes      # PNG bytes of the rasterised page (sent to vision models)
    image_mime_type: str = "image/png"
    text_lines: List[TextLine] = field(default_factory=list)
    embedded_text: Optional[str] = None  # From PDF text layer, if available

    @property
    def full_text(self) -> str:
        """All text on this page as a single string."""
        if self.embedded_text:
            return self.embedded_text
        return "\n".join(line.text for line in self.text_lines)


@dataclass
class OCRDocument:
    """
    Full OCR output for a processed document.

    Passed to providers as context (full_text) and used by attach_bounding_boxes
    to resolve field locations after extraction.
    """
    document_id: str
    pages: List[OCRPage]
    source_mime_type: str

    @property
    def full_text(self) -> str:
        """Concatenated text across all pages, separated by form-feed markers."""
        return "\f".join(page.full_text for page in self.pages)

    @property
    def all_text_lines(self) -> List[TextLine]:
        """Flat list of all TextLines across all pages, in page order."""
        return [line for page in self.pages for line in page.text_lines]

    @property
    def primary_image(self) -> Tuple[bytes, str]:
        """
        Image bytes and MIME type for the first page.
        Providers receive this for vision inference.
        For multi-page documents, providers get page 1 — sufficient for most POs
        where the header fields are on the first page. Line items may span multiple
        pages but totals are typically on the last page too.

        A more sophisticated implementation would send all pages, but for a
        4GB VRAM card and a demo, single-page vision inference is pragmatic.
        """
        if self.pages:
            p = self.pages[0]
            return p.image_bytes, p.image_mime_type
        return b"", "image/png"


# ---------------------------------------------------------------------------
# Surya model singleton — loaded once, reused across all requests
# ---------------------------------------------------------------------------


class _SuryaModels:
    """
    Lazy singleton container for surya 0.17.x predictors.

    surya 0.17.x replaced the old 4-object model API with three predictor classes:
      FoundationPredictor  — shared backbone used by RecognitionPredictor
      DetectionPredictor   — finds text regions (bounding boxes)
      RecognitionPredictor — reads text within detected regions

    Loading takes ~10-15s and consumes ~600MB VRAM on the GTX 1650.
    Must be loaded exactly once and reused across all requests.

    Lock choice: threading.Lock (not asyncio.Lock) so the same singleton
    works safely in both async FastAPI requests and synchronous Celery tasks.
    asyncio.Lock is bound to one event loop — Celery tasks create their own
    loops via asyncio.run(), which would deadlock on an asyncio.Lock.
    """

    def __init__(self) -> None:
        self._foundation = None
        self._det_predictor = None
        self._rec_predictor = None
        self._loaded = False
        self._lock = threading.Lock()  # thread-safe, works in asyncio + Celery

    async def ensure_loaded(self) -> None:
        """Async entry point — delegates blocking load to thread pool."""
        if self._loaded:
            return
        await asyncio.get_running_loop().run_in_executor(
            None, self._ensure_loaded_sync
        )

    def _ensure_loaded_sync(self) -> None:
        """Thread-safe blocking load — called from thread pool."""
        with self._lock:
            if self._loaded:  # double-check after acquiring lock
                return
            self._load_sync()
            self._loaded = True

    def _load_sync(self) -> None:
        """
        Instantiate all three surya 0.17.x predictors.

        surya 0.17.x API (from surya/scripts/ocr_text.py):
            foundation = FoundationPredictor()
            det        = DetectionPredictor(device=device)
            rec        = RecognitionPredictor(foundation)
            results    = rec([image], task_names=[TaskNames.ocr_with_boxes],
                             det_predictor=det, math_mode=False)

        DetectionPredictor accepts a device kwarg.
        FoundationPredictor uses the default device (auto-detects CUDA).
        RecognitionPredictor delegates device to its foundation.
        """
        settings = get_settings()
        device = settings.surya_device

        logger.info(
            "Loading surya 0.17.x predictors on device='%s'. This takes ~15s...",
            device,
        )
        try:
            from surya.foundation import FoundationPredictor   # type: ignore[import]
            from surya.detection import DetectionPredictor     # type: ignore[import]
            from surya.recognition import RecognitionPredictor # type: ignore[import]

            self._foundation     = FoundationPredictor()
            self._det_predictor  = DetectionPredictor(device=device)
            self._rec_predictor  = RecognitionPredictor(self._foundation)

            logger.info(
                "surya predictors loaded successfully on device='%s'.", device
            )
        except ImportError as e:
            logger.error(
                "surya import failed: %s. "
                "Install with: pip install surya-ocr. "
                "OCR will fall back to embedded PDF text extraction only.",
                e,
            )

    @property
    def ready(self) -> bool:
        return self._loaded and self._rec_predictor is not None

    def get_predictors(self):
        """Return (det_predictor, rec_predictor) — used by _run_surya_on_image."""
        return self._det_predictor, self._rec_predictor


# Module-level singleton — shared across all OCRService instances
_surya_models = _SuryaModels()


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class OCRService:
    """
    Processes documents into OCRDocument objects and resolves field bounding boxes.

    Usage in the pipeline:
      1. ocr_doc = await ocr_service.process_document(file_path, mime_type)
      2. [providers run extraction using ocr_doc.full_text and ocr_doc.primary_image]
      3. field_extractions = await ocr_service.attach_bounding_boxes(
             field_extractions, ocr_doc
         )
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Stage 1: Document → OCRDocument
    # ------------------------------------------------------------------

    async def process_document(
        self,
        file_path: str,
        document_id: str,
        mime_type: str,
    ) -> OCRDocument:
        """
        Convert a file on disk into an OCRDocument ready for extraction.

        Fast path (digital PDFs):
          PyMuPDF extracts embedded text in milliseconds. If the extracted text
          is dense enough (> 50 chars/page average), we trust it and skip surya.
          This covers vendor-generated PDFs, which are the majority in practice.

        Slow path (scanned PDFs, images):
          Pages are rasterised to PNG, then surya runs detection + recognition.
          On GPU: ~2s/page. On CPU: ~20s/page.

        Both paths return images — the vision model always gets the rasterised page
        regardless of whether we used embedded text or surya for the text content.
        """
        path = Path(file_path)

        if mime_type == "application/pdf":
            return await self._process_pdf(path, document_id)
        elif mime_type.startswith("image/"):
            return await self._process_image(path, document_id, mime_type)
        else:
            raise ValueError(f"Unsupported MIME type for OCR: {mime_type}")

    async def _process_pdf(self, path: Path, document_id: str) -> OCRDocument:
        """Convert PDF pages to OCRPage objects."""
        pages = await asyncio.get_running_loop().run_in_executor(
            None, self._process_pdf_sync, path, document_id
        )
        return OCRDocument(
            document_id=document_id,
            pages=pages,
            source_mime_type="application/pdf",
        )

    def _process_pdf_sync(self, path: Path, document_id: str) -> List[OCRPage]:
        """
        Blocking PDF processing — runs in thread pool.

        Two paths:
          Digital PDF (embedded text layer):
            → Extract text + bboxes directly from PyMuPDF.
            → More accurate than surya: exact coordinates from the PDF spec,
              not a visual approximation. OCR confidence = 1.0 (ground truth).
            → surya NOT called — avoids VRAM contention with Ollama.

          Scanned PDF (no text layer / sparse text):
            → Rasterise and run surya OCR.
            → surya's detector has issues with some document styles — falls
              back to empty TextLines if detection returns 0 regions.
        """
        try:
            import fitz  # type: ignore[import]  # PyMuPDF
        except ImportError:
            raise RuntimeError(
                "PyMuPDF is not installed. Install with: pip install pymupdf"
            )

        render_dpi = 150.0
        doc = fitz.open(str(path))
        pages: List[OCRPage] = []

        for page_idx in range(len(doc)):
            fitz_page = doc[page_idx]
            page_number = page_idx + 1

            # Rasterise to PNG at 150 DPI — sent to vision model
            mat = fitz.Matrix(render_dpi / 72, render_dpi / 72)
            pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
            image_bytes = pix.tobytes("png")

            # Save page 1 rasterized PNG so the frontend can display it
            # with bounding box overlays (iframes can't have div overlays)
            if page_idx == 0:
                png_path = path.parent / "page_1.png"
                png_path.write_bytes(image_bytes)
                logger.debug("Saved rasterized PNG for frontend: %s", png_path)

            embedded_text = fitz_page.get_text("text").strip()
            is_dense_enough = len(embedded_text) > 50

            ocr_page = OCRPage(
                page_number=page_number,
                image_bytes=image_bytes,
                image_mime_type="image/png",
            )

            if is_dense_enough:
                # Digital PDF — use PyMuPDF text layer for both text and bboxes.
                # PyMuPDF gives us word/line positions in PDF points (72 DPI).
                # Scale by render_dpi/72 to match rasterized image pixel coordinates.
                ocr_page.embedded_text = embedded_text
                ocr_page.text_lines = self._extract_lines_from_pdf_page(
                    fitz_page, page_number, render_dpi=render_dpi
                )
                logger.debug(
                    "Page %d: PyMuPDF bboxes — %d lines extracted from embedded text",
                    page_number, len(ocr_page.text_lines),
                )
            else:
                # Scanned PDF — fall back to surya
                logger.debug(
                    "Page %d: sparse text (%d chars) — running surya OCR",
                    page_number, len(embedded_text),
                )
                ocr_page.text_lines = self._run_surya_on_image(image_bytes, page_number)

            pages.append(ocr_page)

        doc.close()
        return pages

    @staticmethod
    def _extract_lines_from_pdf_page(
        fitz_page,
        page_number: int,
        render_dpi: float = 150.0,
    ) -> List[TextLine]:
        """
        Extract text with precise bboxes from a PyMuPDF page.

        Two levels of extraction for maximum matching coverage:

        1. Word-level (primary): each word gets its own tight bbox.
           Fixes the "stacked on top of each other" problem from block
           splitting — words at the same Y but different X don't overlap.

        2. Line-level (secondary): adjacent words on the same Y are grouped
           into compound TextLines. Enables matching multi-word values like
           "Roland Mendel" or "Acme Corporation" as a single bbox.

        Uses get_text('rawdict') which gives precise character-level positions
        baked into line and word objects — more accurate than block splitting.
        """
        scale = render_dpi / 72.0
        text_lines: List[TextLine] = []
        seen_texts: set = set()  # avoid duplicate TextLines

        def add_line(text: str, x0, y0, x1, y1):
            text = text.strip()
            if not text:
                return
            key = f"{text}:{round(x0)}:{round(y0)}"
            if key in seen_texts:
                return
            seen_texts.add(key)
            text_lines.append(TextLine(
                text=text,
                x0=x0 * scale, y0=y0 * scale,
                x1=x1 * scale, y1=y1 * scale,
                page=page_number,
                ocr_confidence=1.0,
            ))

        # ── Word-level extraction ────────────────────────────────────────
        words = fitz_page.get_text('words')
        # words: (x0, y0, x1, y1, text, block_no, line_no, word_no)
        for w in words:
            add_line(w[4], w[0], w[1], w[2], w[3])

        # ── Line-level extraction via dict ───────────────────────────────
        # Groups adjacent words on the same physical line into compound TextLines
        # so multi-word values ("Roland Mendel") can be matched as one bbox
        try:
            page_dict = fitz_page.get_text('dict')
            for block in page_dict.get('blocks', []):
                if block.get('type') != 0:  # 0 = text
                    continue
                for line in block.get('lines', []):
                    # Concatenate all spans in this line
                    line_text = ' '.join(
                        span.get('text', '') for span in line.get('spans', [])
                    ).strip()
                    if not line_text:
                        continue
                    bbox = line.get('bbox')
                    if bbox and len(bbox) == 4:
                        add_line(line_text, bbox[0], bbox[1], bbox[2], bbox[3])
        except Exception as e:
            logger.debug("Line-level PDF extraction failed: %s", e)

        logger.debug(
            "PyMuPDF bboxes: page %d → %d text elements",
            page_number, len(text_lines),
        )
        return text_lines

    async def _process_image(
        self, path: Path, document_id: str, mime_type: str
    ) -> OCRDocument:
        """Process a single image file."""
        image_bytes = path.read_bytes()

        text_lines = await asyncio.get_running_loop().run_in_executor(
            None, self._run_surya_on_image, image_bytes, 1
        )

        page = OCRPage(
            page_number=1,
            image_bytes=image_bytes,
            image_mime_type=mime_type,
            text_lines=text_lines,
        )
        return OCRDocument(
            document_id=document_id,
            pages=[page],
            source_mime_type=mime_type,
        )

    def _run_surya_on_image(
        self, image_bytes: bytes, page_number: int
    ) -> List[TextLine]:
        """
        Run surya OCR on one page image and return TextLine objects.

        Two strategies:
          1. Detection-based (preferred): run det_predictor to find text regions,
             then run recognition on those regions. More accurate bboxes.
             Falls back to strips if detector returns 0 regions (known issue
             with some document styles).

          2. Strip-based fallback: divide the image into horizontal strips,
             pass each as a bbox to recognition — bypasses the broken detector.
             Used for scanned documents where detection fails.
             Bboxes are approximate (strip-level) but good enough for matching.
        """
        if not _surya_models.ready:
            logger.warning(
                "surya predictors not loaded — skipping OCR for page %d.",
                page_number,
            )
            return []

        try:
            import io
            from PIL import Image                              # type: ignore[import]
            from surya.common.surya.schema import TaskNames   # type: ignore[import]

            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            det_predictor, rec_predictor = _surya_models.get_predictors()

            # ── Strategy 1: detection-based ──────────────────────────────
            results = rec_predictor(
                [pil_image],
                task_names=[TaskNames.ocr_with_boxes],
                det_predictor=det_predictor,
                math_mode=False,
            )

            if results and len(results[0].text_lines) > 0:
                text_lines = self._parse_surya_text_lines(
                    results[0].text_lines, page_number
                )
                logger.debug(
                    "surya OCR (detection): page %d → %d text lines",
                    page_number, len(text_lines),
                )
                return text_lines

            # ── Strategy 2: strip-based fallback ─────────────────────────
            logger.info(
                "surya detector found 0 regions on page %d — "
                "falling back to horizontal strip OCR (scanned document).",
                page_number,
            )
            return self._run_surya_strips(
                pil_image, rec_predictor, page_number
            )

        except Exception as e:
            logger.error(
                "surya OCR failed on page %d: %s", page_number, e, exc_info=True
            )
            return []

    def _run_surya_strips(
        self,
        pil_image,
        rec_predictor,
        page_number: int,
        strip_height: int = 80,
    ) -> List[TextLine]:
        """
        OCR a scanned image by passing horizontal strips as bboxes.

        Bypasses surya's detection model (which fails on many scanned documents)
        by manually specifying regions. Each strip is a horizontal band across
        the full image width.

        strip_height=80px at 150 DPI covers roughly 1-2 lines of typical
        PO text. Adjust lower (50px) for dense documents, higher (100px) for
        sparse ones.

        The resulting TextLine bboxes are strip-level approximations —
        not pixel-perfect, but accurate enough for the fuzzy matching in
        attach_bounding_boxes to find the right region.
        """
        from surya.common.surya.schema import TaskNames  # type: ignore[import]

        w, h = pil_image.size

        # Build horizontal strip bboxes covering the full page
        strips = [
            [0, y, w, min(y + strip_height, h)]
            for y in range(0, h, strip_height)
        ]

        try:
            results = rec_predictor(
                [pil_image],
                task_names=[TaskNames.ocr_with_boxes],
                bboxes=[strips],   # ← bypass detection, provide regions manually
                math_mode=False,
            )
        except Exception as e:
            logger.error("Strip OCR failed: %s", e, exc_info=True)
            return []

        if not results:
            return []

        text_lines = self._parse_surya_text_lines(
            results[0].text_lines, page_number
        )

        logger.debug(
            "surya OCR (strips): page %d → %d strips → %d text lines",
            page_number, len(strips), len(text_lines),
        )
        return text_lines

    @staticmethod
    def _parse_surya_text_lines(surya_lines, page_number: int) -> List[TextLine]:
        """
        Convert surya OCRResult text_lines to our TextLine dataclass.
        Shared by both detection-based and strip-based strategies.
        """
        text_lines: List[TextLine] = []
        for surya_line in surya_lines:
            bbox = surya_line.bbox
            if not bbox or len(bbox) != 4:
                continue
            text = getattr(surya_line, "text", "").strip()
            if not text:
                continue
            text_lines.append(TextLine(
                text=text,
                x0=float(bbox[0]),
                y0=float(bbox[1]),
                x1=float(bbox[2]),
                y1=float(bbox[3]),
                page=page_number,
                ocr_confidence=getattr(surya_line, "confidence", 0.85),
            ))
        return text_lines

    # ------------------------------------------------------------------
    # Stage 2: Attach bounding boxes — three-layer matching
    # ------------------------------------------------------------------

    async def attach_bounding_boxes(
        self,
        field_extractions: Dict[str, FieldExtraction],
        ocr_doc: OCRDocument,
    ) -> Dict[str, FieldExtraction]:
        """
        Resolve the pixel location of each extracted field value in the document.

        For each field in field_extractions:
          1. Skip if value is None (nothing to locate)
          2. Try Layer 1: exact string match against surya text lines
          3. Try Layer 2: fuzzy normalised match (rapidfuzz)
          4. Try Layer 3: LLM spatial hint via Ollama (async, costs one inference)
          5. If all fail: leave bounding_box=None, set OCRMatchMethod.UNRESOLVED

        Returns an updated copy of field_extractions with bounding_box populated
        where resolution succeeded.
        """
        all_lines = ocr_doc.all_text_lines
        updated: Dict[str, FieldExtraction] = {}

        for field_path, extraction in field_extractions.items():
            if extraction.value is None:
                updated[field_path] = extraction
                continue

            # Skip list/dict values (line_items) — too complex for simple matching
            if not isinstance(extraction.value, (str, int, float)):
                updated[field_path] = extraction
                continue

            value_str = str(extraction.value).strip()
            if len(value_str) < 2:
                updated[field_path] = extraction
                continue

            bbox, method = self._resolve_bounding_box(value_str, all_lines)

            # Layer 3 (LLM hint) is async — only call if layers 1 and 2 failed
            if bbox is None and all_lines:
                bbox, method = await self._llm_spatial_hint(
                    field_path, value_str, all_lines
                )

            if bbox is not None:
                updated_extraction = extraction.model_copy(
                    update={"bounding_box": bbox.model_copy(update={"ocr_match_method": method})}
                )
                logger.debug(
                    "Field '%s': bounding box resolved via %s", field_path, method.value
                )
            else:
                updated_extraction = extraction
                logger.debug(
                    "Field '%s': bounding box unresolved (value=%r)", field_path, value_str[:40]
                )

            updated[field_path] = updated_extraction

        return updated

    def _resolve_bounding_box(
        self,
        value_str: str,
        all_lines: List[TextLine],
    ) -> Tuple[Optional[BoundingBox], OCRMatchMethod]:
        """
        Run Layer 1 then Layer 2. Returns (bbox, method) or (None, UNRESOLVED).
        Layer 3 is async and handled by the caller.
        """
        # Layer 1: Exact match
        bbox = self._exact_match(value_str, all_lines)
        if bbox is not None:
            return bbox, OCRMatchMethod.EXACT

        # Layer 2: Fuzzy normalised match
        bbox = self._fuzzy_match(value_str, all_lines)
        if bbox is not None:
            return bbox, OCRMatchMethod.FUZZY

        return None, OCRMatchMethod.UNRESOLVED

    # ------------------------------------------------------------------
    # Layer 1 — Exact match
    # ------------------------------------------------------------------

    @staticmethod
    def _exact_match(
        value_str: str, all_lines: List[TextLine]
    ) -> Optional[BoundingBox]:
        """
        Find the best-matching text line for value_str.

        Matching priority (highest wins):
          1. Line text exactly equals value (ratio = 1.0) — e.g. "Laurence Lebihan"
          2. Line contains value AND line is close in length to value
             ratio = len(value) / len(line) — prefers tighter bboxes
          3. Value contains line (value spans multiple OCR lines)
             ratio = len(line) / len(value)

        This prevents "Laurence" from winning over "Laurence Lebihan" because:
          - "Laurence Lebihan" in "Laurence Lebihan" → ratio = 1.0 ✓
          - "Laurence Lebihan" in "11076 2018-05-06 Laurence Lebihan" → ratio = 0.47
          - "Laurence" in "Laurence Lebihan" → ratio = 0.53 (loses to exact match)
        """
        value_lower = value_str.lower().strip()
        if not value_lower:
            return None

        best_line: Optional[TextLine] = None
        best_ratio: float = 0.0

        for line in all_lines:
            line_lower = line.text.lower().strip()
            if not line_lower:
                continue

            # Exact match — highest priority
            if line_lower == value_lower:
                return line.to_bounding_box(OCRMatchMethod.EXACT)

            # Line contains value — ratio = how much of the line is the value
            if value_lower in line_lower:
                ratio = len(value_lower) / len(line_lower)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_line = line

            # Value contains line — for multi-word values split across OCR lines
            elif line_lower in value_lower and len(line_lower) > 2:
                ratio = len(line_lower) / len(value_lower)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_line = line

        return best_line.to_bounding_box(OCRMatchMethod.EXACT) if best_line else None

    # ------------------------------------------------------------------
    # Layer 2 — Fuzzy normalised match
    # ------------------------------------------------------------------

    def _fuzzy_match(
        self,
        value_str: str,
        all_lines: List[TextLine],
    ) -> Optional[BoundingBox]:
        """
        Normalise both value and OCR text, then find the best fuzzy match.

        Normalisation handles common OCR degradation patterns:
          - Case differences ("ACME" vs "Acme")
          - Punctuation noise ("Corp." vs "Corp")
          - Whitespace collapse ("New  York" vs "New York")
          - Unicode normalisation ("café" vs "cafe")

        Uses rapidfuzz.fuzz.partial_ratio — scores a substring match rather than
        full-string similarity. This handles the common case where surya returns
        a line like "Vendor: Acme Corporation" and we're looking for "Acme Corporation".
        """
        try:
            from rapidfuzz import fuzz  # type: ignore[import]
        except ImportError:
            logger.warning(
                "rapidfuzz not installed — Layer 2 fuzzy match unavailable. "
                "Install with: pip install rapidfuzz"
            )
            return None

        threshold = self._settings.ocr_fuzzy_match_threshold * 100  # 0–100 scale
        normalised_value = _normalise(value_str)

        if len(normalised_value) < 3:
            return None  # Too short to fuzzy match reliably

        best_score = 0.0
        best_line: Optional[TextLine] = None

        for line in all_lines:
            normalised_line = line.normalised_text
            if not normalised_line:
                continue

            score = fuzz.partial_ratio(normalised_value, normalised_line)
            if score > best_score:
                best_score = score
                best_line = line

        if best_score >= threshold and best_line is not None:
            logger.debug(
                "Fuzzy match: '%s' → '%s' (score=%.1f)",
                value_str[:30], best_line.text[:30], best_score
            )
            return best_line.to_bounding_box(OCRMatchMethod.FUZZY)

        return None

    # ------------------------------------------------------------------
    # Layer 3 — LLM spatial hint (last resort, async)
    # ------------------------------------------------------------------

    async def _llm_spatial_hint(
        self,
        field_name: str,
        value_str: str,
        all_lines: List[TextLine],
    ) -> Tuple[Optional[BoundingBox], OCRMatchMethod]:
        """
        Ask the local Ollama model which numbered text line contains the target field.

        Why Ollama, not Claude?
          This is a spatial lookup question, not a quality-sensitive extraction.
          Using Claude here would incur API cost on every bounding box miss.
          Ollama runs locally at no cost. If Ollama is unavailable, we return
          (None, UNRESOLVED) and the UI shows the field without a highlight.

        Approach:
          1. Build a numbered list of OCR text lines (max 100 — most POs fit)
          2. Ask Ollama: "which index contains [field_name] = [value]?"
          3. Parse the response as {"line_index": N}
          4. Return that line's bounding box

        The prompt is intentionally minimal — one small fast inference.
        """
        settings = get_settings()
        ollama_url = f"{settings.ollama_base_url}/api/generate"

        # Build numbered line list (cap at 100 lines to keep prompt short)
        lines_sample = all_lines[:100]
        numbered_lines = "\n".join(
            f"{i}: {line.text}" for i, line in enumerate(lines_sample)
        )

        prompt = (
            f"Given these numbered text lines from a purchase order document:\n\n"
            f"{numbered_lines}\n\n"
            f"Which line index (0-based) most likely contains the value for "
            f"field '{field_name}' with value '{value_str}'?\n"
            f"Respond with ONLY valid JSON: {{\"line_index\": <integer or null>}}\n"
            f"Use null if you cannot find it."
        )

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    ollama_url,
                    json={
                        "model": settings.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0, "num_predict": 32},
                    },
                )

            if response.status_code != 200:
                logger.debug(
                    "LLM spatial hint: Ollama returned HTTP %d", response.status_code
                )
                return None, OCRMatchMethod.UNRESOLVED

            import json
            raw = response.json().get("response", "")

            # Parse the response — expects {"line_index": N}
            cleaned = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not cleaned:
                return None, OCRMatchMethod.UNRESOLVED

            parsed = json.loads(cleaned.group())
            line_index = parsed.get("line_index")

            if line_index is None or not isinstance(line_index, int):
                return None, OCRMatchMethod.UNRESOLVED

            if 0 <= line_index < len(lines_sample):
                matched_line = lines_sample[line_index]
                logger.debug(
                    "LLM spatial hint: field='%s' → line %d '%s'",
                    field_name, line_index, matched_line.text[:40]
                )
                return (
                    matched_line.to_bounding_box(OCRMatchMethod.LLM_HINT),
                    OCRMatchMethod.LLM_HINT,
                )

        except httpx.ConnectError:
            logger.debug("LLM spatial hint: Ollama unavailable — skipping Layer 3")
        except httpx.TimeoutException:
            logger.debug("LLM spatial hint: Ollama timed out — skipping Layer 3")
        except Exception as e:
            logger.debug("LLM spatial hint failed: %s", e)

        return None, OCRMatchMethod.UNRESOLVED

    # ------------------------------------------------------------------
    # Startup warm-up
    # ------------------------------------------------------------------

    async def warm_up(self) -> None:
        """
        Pre-load surya models at application startup.

        Called from main.py lifespan handler so the first real document request
        doesn't pay the 10s model loading penalty. Safe to call multiple times.
        """
        await _surya_models.ensure_loaded()
        logger.info("OCR service warm-up complete. surya_ready=%s", _surya_models.ready)


# ---------------------------------------------------------------------------
# Text normalisation — shared by Layer 1 and Layer 2
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """
    Normalise a string for fuzzy matching.

    Steps:
      1. Unicode NFKD normalisation (é → e + combining accent)
      2. Strip combining characters (drop the accent)
      3. Lowercase
      4. Collapse all whitespace to single space
      5. Remove punctuation except hyphens and slashes (relevant in PO numbers)

    Examples:
      "ACME Corp."       → "acme corp"
      "Café & Co."       → "cafe co"
      "P.O. #12-345/A"   → "po 12-345/a"
      "New  York  City"  → "new york city"
    """
    # Unicode normalisation — strip accents
    normalised = unicodedata.normalize("NFKD", text)
    normalised = "".join(c for c in normalised if not unicodedata.combining(c))

    # Lowercase
    normalised = normalised.lower()

    # Remove punctuation except hyphens and forward slashes
    normalised = re.sub(r"[^\w\s\-/]", " ", normalised)

    # Collapse whitespace
    normalised = re.sub(r"\s+", " ", normalised).strip()

    return normalised