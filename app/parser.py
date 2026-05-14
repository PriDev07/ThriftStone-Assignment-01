"""
PDF parsing with pymupdf4llm.

Why pymupdf4llm:
  Converts each PDF page to structured Markdown, preserving slide hierarchy:
  headings, bold numbers, bullet lists, and tables as Markdown pipe tables.
  Bold KPI values (e.g. "**1101**") stay adjacent to their label text on the
  same line, so the KPI normaliser in chunker.py can pair them reliably.
  Page numbers come from chunk["metadata"]["page_number"] (1-based).

OCR fallback (pdf2image + pytesseract):
  Triggered when pymupdf4llm returns fewer than OCR_TRIGGER_CHARS
  non-whitespace characters for a page (image-only slides). Merged with
  word-overlap deduplication. ocr_used flag stored per page.

Where it still fails:
  - Scanned PDFs with no embedded text layer still need OCR.
  - Complex infographic charts (bar/pie) with no text cannot be parsed.
  - Rotated or heavily stylised fonts may produce garbled OCR output.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple

import pymupdf4llm

# OCR imports — optional; gracefully disabled if not installed
try:
    from pdf2image import convert_from_path
    import pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

# Trigger OCR when fewer than this many non-whitespace characters are extracted
OCR_TRIGGER_CHARS = 80


# ---------------------------------------------------------------------------
# Section title tagging
# ---------------------------------------------------------------------------

def _tag_section_title(md_text: str) -> str:
    """
    Derive a canonical section tag from the page's Markdown content.

    Rules are checked in priority order:

    1. "Results Context" / "RESULTS CONTEXT"     → ``Results Context``
    2. "AT A GLANCE" / "HIGHLIGHTS"              → ``Key Metrics``
    3. "TRENDS" / "COMPARATIVE PERIODS"          → ``Financial Trends``
    4. "CAMPAIGNS" / "CONNECT" / "RECENT STORES" → ``Brand Activity``
    5. "SUSTAINABILITY" / "CSR"                  → ``Sustainability``
    6. Commentary signals (headwind, risk, …)    → ``Management Commentary``
    7. First Markdown heading (``# …``)          → heading text (≤ 120 chars)
    8. Financial signals (revenue, ebitda, …)    → ``Financial Results``
    9. Fallback                                  → ``General``

    Parameters
    ----------
    md_text : str
        Cleaned Markdown text for one PDF page.

    Returns
    -------
    str
        Canonical section tag.
    """
    upper = md_text.upper()

    if "RESULTS CONTEXT" in upper:
        return "Results Context"
    if "AT A GLANCE" in upper or "HIGHLIGHTS" in upper:
        return "Key Metrics"
    if "TRENDS" in upper or "COMPARATIVE PERIODS" in upper:
        return "Financial Trends"
    if "CAMPAIGNS" in upper or "CONNECT" in upper or "RECENT STORES" in upper:
        return "Brand Activity"
    if "SUSTAINABILITY" in upper or "CSR" in upper:
        return "Sustainability"

    lower = md_text.lower()
    commentary_signals = [
        "headwind", "tailwind", "sentiment", "macro", "context",
        "challenge", "risk", "outlook", "management comment",
        "operating environment", "consumer demand",
        "inflationary", "discretionary", "footfall",
    ]
    if any(sig in lower for sig in commentary_signals):
        return "Management Commentary"

    # First Markdown heading takes priority over the broad financial heuristic
    # so that a page titled "# Revenue Overview" gets its heading as the tag.
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = re.sub(r"^#+\s*", "", stripped).strip()
            if heading:
                return heading[:120]

    financial_signals = [
        "revenue", "ebitda", "pat", "margin", "profit", "loss",
        "growth", "yoy", "quarter", "fy", "crore", "lakh",
    ]
    if any(sig in lower for sig in financial_signals):
        return "Financial Results"

    return "General"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_md(raw: str) -> str:
    """
    Lightly normalise pymupdf4llm Markdown output.

    Collapses runs of 3+ blank lines to 2 and strips trailing whitespace
    from each line. Does not alter Markdown syntax (tables, bold, headings).

    Parameters
    ----------
    raw : str
        Raw Markdown string from pymupdf4llm.

    Returns
    -------
    str
        Cleaned Markdown string.
    """
    text = re.sub(r"\n{3,}", "\n\n", raw)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _merge_ocr_text(primary_text: str, ocr_text: str) -> str:
    """
    Merge primary (pymupdf4llm) and OCR text, keeping only OCR lines that
    are not already substantially present in the primary output.

    A line is considered a duplicate if more than 60% of its words already
    appear in the primary text.

    Parameters
    ----------
    primary_text : str
        Text extracted by pymupdf4llm.
    ocr_text : str
        Text extracted by Tesseract OCR.

    Returns
    -------
    str
        Merged text with deduplicated lines.
    """
    existing_words = set(re.findall(r"\w+", primary_text.lower()))
    merged_lines = list(primary_text.splitlines())

    for line in ocr_text.splitlines():
        line = line.strip()
        if not line:
            continue
        line_words = set(re.findall(r"\w+", line.lower()))
        if not line_words:
            continue
        overlap = len(line_words & existing_words) / len(line_words)
        if overlap < 0.6:
            merged_lines.append(line)
            existing_words |= line_words

    return "\n".join(merged_lines)


def _ocr_page(pdf_path: Path, page_number: int) -> str:
    """
    Render a single PDF page to an image and run Tesseract OCR on it.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.
    page_number : int
        1-based page number to render.

    Returns
    -------
    str
        OCR text, or empty string if OCR is unavailable or fails.
    """
    if not _OCR_AVAILABLE:
        return ""
    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=200,
            first_page=page_number,
            last_page=page_number,
        )
        if not images:
            return ""
        return pytesseract.image_to_string(images[0], config="--psm 6")
    except Exception:
        logger.warning(
            "OCR failed for page %d of %s", page_number, pdf_path, exc_info=True
        )
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str | Path) -> List[Tuple[int, str, str, bool]]:
    """
    Parse a PDF with pymupdf4llm and return one tuple per page.

    Each tuple contains:
      ``(page_number, section_title, cleaned_markdown_text, ocr_used)``

    Page numbers are 1-indexed. ``ocr_used`` is True when the Tesseract
    fallback contributed text to the page.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file.

    Returns
    -------
    list of (int, str, str, bool)
        One entry per non-empty page.

    Raises
    ------
    FileNotFoundError
        If the PDF does not exist at the given path.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # pymupdf4llm >= 0.0.17 stores the 1-based page number in
    # chunk["metadata"]["page_number"]. The top-level "page" key is None.
    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    pages: List[Tuple[int, str, str, bool]] = []

    for chunk in page_chunks:
        meta = chunk.get("metadata") or {}
        page_number = meta.get("page_number") or (int(chunk.get("page") or 0) + 1)
        raw_md = chunk.get("text", "") or ""

        # OCR fallback for pages with very little extractable text
        ocr_used = False
        if len(re.sub(r"\s", "", raw_md)) < OCR_TRIGGER_CHARS:
            ocr_text = _ocr_page(pdf_path, page_number)
            if ocr_text.strip():
                raw_md = _merge_ocr_text(raw_md, ocr_text)
                ocr_used = True

        cleaned = _clean_md(raw_md)
        if not cleaned:
            continue

        section_title = _tag_section_title(cleaned)
        pages.append((page_number, section_title, cleaned, ocr_used))

    return pages
