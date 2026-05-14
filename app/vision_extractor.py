"""
Vision-based content extractor for image-heavy PDF slides.

Investor presentations commonly render key data as bar charts, logo images,
and KPI infographics rather than selectable text. This module detects pages
where text extraction is insufficient and falls back to a vision LLM
(llama-4-scout-17b-16e-instruct) to describe the visual content in structured
text format.

This is called automatically during ingestion for any page where meaningful
text content is below the minimum word threshold, or where chart indicator
keywords are present but the actual numeric values are missing.
"""
from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any

from groq import Groq

# pdf2image is imported at module level so it can be patched in tests.
# Gracefully set to None if not installed; extract_page_as_base64 checks this.
try:
    from pdf2image import convert_from_path as _convert_from_path
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    _convert_from_path = None  # type: ignore[assignment]
    _PDF2IMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SPARSE_PAGE_MIN_WORDS = 30
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
VISION_MAX_TOKENS = 1000

_VISION_SYSTEM_PROMPT = (
    "You are analyzing a slide from an investor presentation PDF.\n"
    "Extract ALL visible information precisely.\n"
    "Return your response in this exact format:\n\n"
    "SLIDE_TOPIC: <one line describing what this slide is about>\n"
    "BRAND: <brand or company name visible on slide, or NONE>\n"
    "KEY_METRICS:\n"
    "- <exact label>: <exact value with units>\n"
    "- <exact label>: <exact value with units>\n"
    "CHART_DATA:\n"
    "- <x-axis label> : <y-axis value>\n"
    "- <x-axis label> : <y-axis value>\n"
    "OTHER_TEXT:\n"
    "<any other text visible on the slide>\n\n"
    "RULES:\n"
    "- Never guess or hallucinate numbers\n"
    "- Copy numbers exactly as they appear\n"
    "- Include units (Cr, Mn, %, sq ft, etc.)\n"
    "- If a section has nothing, write NONE under it\n"
    "- For bar charts, list every single bar with its label and value"
)

# Chart indicator keywords: if any of these appear in the extracted text but
# no 3-digit number is present, the page is considered structurally incomplete.
_CHART_INDICATORS = [
    "Store Count", "Store count", "store count",
    "Revenue", "revenue",
    "Q2 FY22", "Q2FY22", "Q2 FY23", "Q2FY23",
    "Q2 FY24", "Q2FY24", "Q2 FY25", "Q2FY25",
    "Q2 FY26", "Q2FY26",
]

# ---------------------------------------------------------------------------
# Lazy Groq client
# ---------------------------------------------------------------------------

_client_instance: Groq | None = None


def _get_client() -> Groq:
    """Return the module-level Groq client, creating it on first call."""
    global _client_instance
    if _client_instance is None:
        _client_instance = Groq()
    return _client_instance


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def extract_page_as_base64(pdf_path: str, page_number: int) -> str:
    """
    Render a single PDF page to a PNG image and return it as a base64 string.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file.
    page_number : int
        1-based page number to render.

    Returns
    -------
    str
        Base64-encoded PNG image string, or empty string on any failure.
    """
    try:
        if not _PDF2IMAGE_AVAILABLE or _convert_from_path is None:
            logger.warning(
                "pdf2image is not installed; cannot render page %d of %s",
                page_number, pdf_path,
            )
            return ""

        images = _convert_from_path(
            pdf_path,
            dpi=150,
            first_page=page_number,
            last_page=page_number,
        )
        if not images:
            logger.warning(
                "pdf2image returned no images for page %d of %s",
                page_number, pdf_path,
            )
            return ""

        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception:
        logger.warning(
            "extract_page_as_base64 failed for page %d of %s",
            page_number, pdf_path, exc_info=True,
        )
        return ""


def is_content_sparse(text: str) -> bool:
    """
    Return True if the extracted text is too sparse or structurally incomplete
    to be useful for retrieval without vision augmentation.

    Two conditions trigger sparsity (either is sufficient):

    1. Fewer than ``SPARSE_PAGE_MIN_WORDS`` words remain after stripping
       pymupdf4llm image placeholder lines (``==> ... <==`` and
       ``----- ... -----``).
    2. The text contains a chart indicator keyword (e.g. "Store Count",
       "Revenue", "Q2 FY22") but contains no number with 3+ digits — meaning
       axis labels were extracted but bar values are embedded in images.

    Parameters
    ----------
    text : str
        Raw text extracted from a PDF page.

    Returns
    -------
    bool
        True if vision fallback is needed, False otherwise.
    """
    # Strip pymupdf4llm placeholder lines line-by-line (not with DOTALL) to
    # avoid accidentally consuming real content between separator markers.
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # "==> image.png <==" style image placeholders
        if re.match(r"^==>.*<==\s*$", stripped):
            continue
        # "----- separator -----" style dividers (3+ dashes on each side)
        if re.match(r"^-{3,}.*-{3,}\s*$", stripped):
            continue
        cleaned_lines.append(stripped)

    cleaned = " ".join(cleaned_lines)
    word_count = len(cleaned.split())

    # Condition 1: too few words after cleaning
    if word_count < SPARSE_PAGE_MIN_WORDS:
        return True

    # Condition 2: chart indicator present but numeric values are absent.
    # Only evaluated when word_count >= SPARSE_PAGE_MIN_WORDS so that
    # condition 1 is not double-counted.
    has_chart_indicator = any(kw in cleaned for kw in _CHART_INDICATORS)
    has_large_numbers = bool(re.search(r"\b\d{3,}\b", cleaned))

    if has_chart_indicator and not has_large_numbers:
        return True

    return False


def describe_page_with_vision(image_b64: str) -> str:
    """
    Send a base64-encoded page image to the Groq vision API and return a
    structured text description.

    Parameters
    ----------
    image_b64 : str
        Base64-encoded PNG image of the PDF page.

    Returns
    -------
    str
        Structured description from the vision model, or empty string on
        any failure.
    """
    if not image_b64:
        return ""

    try:
        data_url = f"data:image/png;base64,{image_b64}"
        response = _get_client().chat.completions.create(
            model=VISION_MODEL,
            max_tokens=VISION_MAX_TOKENS,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": _VISION_SYSTEM_PROMPT},
                    ],
                }
            ],
        )
        return response.choices[0].message.content or ""

    except Exception:
        logger.warning("describe_page_with_vision failed", exc_info=True)
        return ""


def extract_visual_content(
    pdf_path: str,
    page_number: int,
    existing_text: str,
) -> dict[str, Any]:
    """
    Main entry point called by the ingestion pipeline for each PDF page.

    Checks whether the existing text is sparse; if so, renders the page as
    an image and calls the Groq vision API to extract structured content.
    The vision description is prepended to any existing text.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    page_number : int
        1-based page number.
    existing_text : str
        Text already extracted by pymupdf4llm for this page.

    Returns
    -------
    dict with exactly these keys:
        ``final_text``        — merged or original text
        ``used_vision``       — whether the vision API was called
        ``word_count_before`` — word count of existing_text before vision
    """
    word_count_before = len(existing_text.split())

    if not is_content_sparse(existing_text):
        return {
            "final_text": existing_text,
            "used_vision": False,
            "word_count_before": word_count_before,
        }

    try:
        image_b64 = extract_page_as_base64(pdf_path, page_number)
    except Exception:
        logger.warning(
            "extract_page_as_base64 raised unexpectedly for page %d",
            page_number, exc_info=True,
        )
        image_b64 = ""

    if not image_b64:
        return {
            "final_text": existing_text,
            "used_vision": False,
            "word_count_before": word_count_before,
        }

    vision_description = describe_page_with_vision(image_b64)
    if not vision_description:
        return {
            "final_text": existing_text,
            "used_vision": False,
            "word_count_before": word_count_before,
        }

    # Vision description comes first; existing text appended below a separator
    parts = [vision_description.strip()]
    if existing_text.strip():
        parts.append("--- TEXT EXTRACTION ---")
        parts.append(existing_text.strip())

    return {
        "final_text": "\n".join(parts),
        "used_vision": True,
        "word_count_before": word_count_before,
    }
