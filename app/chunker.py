"""
Chunking strategy.

Strategy: KPI-aware, sentence-boundary chunking per page.

Rationale:
- Investor presentations are slide-based; each page is already a logical unit.
- We first try to keep each page as a single chunk (most slides are short).
- If a page exceeds `chunk_size` words, we split on sentence boundaries rather
  than raw word windows, so label-value pairs (e.g. "Zudio: 806 stores") are
  never split across chunk boundaries.
- KPI blocks — contiguous groups of lines where each line contains both a
  number/percentage and a label — are detected and treated as atomic units
  that cannot be split. This prevents the LLM from reassigning a metric from
  one brand/entity to another.
- Before chunking, a KPI normalization pass rewrites raw pdfplumber output like
  "1101 Store count 251 City Presence" into explicit key-value sentences:
  "Store count: 1101. City Presence: 251." This happens before embedding so
  the stored chunk text is always in the normalized form.
- Margin and growth percentages on the same page are explicitly labelled
  ("Margin: X%" vs "YoY Growth: X%") during a pre-processing pass so the LLM
  can distinguish them.
- Tables extracted by pdfplumber are kept inline so financial rows stay
  together with their headers.

Chunk size: 400 words (~500-600 tokens).
Overlap: 80 words — applied only to non-KPI prose splits.

Trade-offs:
- KPI blocks that exceed chunk_size are kept whole (no split), which may
  produce chunks larger than the nominal limit. This is intentional — accuracy
  of label-value association outweighs size uniformity.
- Very sparse pages (title slides) produce tiny chunks — acceptable because
  they carry little retrieval value.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from app.models import Chunk

# ---------------------------------------------------------------------------
# Legacy word-window splitter (kept for backward compatibility and tests)
# ---------------------------------------------------------------------------

def _word_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping word windows (naive, no KPI awareness)."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# KPI block detection
# ---------------------------------------------------------------------------

# Matches a line that contains at least one number (integer, decimal, or
# comma-separated) and at least one alphabetic word — i.e. a "label: value"
# or "value label" pattern typical of KPI slides, including the normalized
# "Label: 1,101." form produced by _normalize_kpi_line.
# Matches either number-first ("1,101 Stores") or label-first ("Stores 1,101").
_KPI_LINE_RE = re.compile(
    r"(?:"
    r"(?:[\d,]+(?:\.\d+)?%?)\s+[A-Za-z]{2,}"   # number then word
    r"|"
    r"[A-Za-z]{2,}.*[\d,]+(?:\.\d+)?%?"         # word then number
    r")",
    re.IGNORECASE,
)

# Matches a standalone percentage that looks like a margin (e.g. "28%", "10.5%")
_MARGIN_RE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*%")

# Matches YoY / QoQ growth patterns (e.g. "+12%", "grew 8%", "growth of 15%")
_GROWTH_RE = re.compile(
    r"(?:yoy|qoq|year.on.year|quarter.on.quarter|grew?|growth(?: of)?|"
    r"increase[d]?(?: of| by)?|decline[d]?(?: of| by)?|"
    r"up|down)\s*(?:by\s*)?([+-]?\d{1,3}(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def _is_kpi_line(line: str) -> bool:
    """Return True if the line looks like a KPI label-value pair."""
    return bool(_KPI_LINE_RE.search(line))


def _detect_kpi_blocks(lines: List[str], min_block_size: int = 2) -> List[Tuple[int, int]]:
    """
    Identify contiguous runs of KPI lines.

    Returns a list of (start_idx, end_idx) pairs (inclusive) for each block
    that has at least `min_block_size` consecutive KPI lines.
    """
    blocks: List[Tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if _is_kpi_line(lines[i]):
            j = i
            while j < len(lines) and _is_kpi_line(lines[j]):
                j += 1
            if j - i >= min_block_size:
                blocks.append((i, j - 1))
            i = j
        else:
            i += 1
    return blocks


def _label_margin_vs_growth(text: str) -> str:
    """
    Pre-processing pass: explicitly label margin percentages and YoY growth
    percentages so the LLM can distinguish them on slides that show both.

    Strategy:
    - Lines containing growth/YoY keywords get "YoY Growth:" prepended to
      the percentage token.
    - Remaining standalone percentages on financial result pages get
      "Margin:" prepended.

    This is a best-effort heuristic; it does not modify numbers, only adds
    clarifying labels.
    """
    result_lines = []
    for line in text.splitlines():
        # If the line already has an explicit growth keyword, tag the %
        if _GROWTH_RE.search(line):
            line = _GROWTH_RE.sub(
                lambda m: m.group(0).replace(
                    m.group(1) + "%", f"YoY Growth: {m.group(1)}%"
                ),
                line,
            )
        elif _MARGIN_RE.search(line) and any(
            kw in line.lower()
            for kw in ("margin", "ebitda", "pat", "profit", "return", "roce", "roe")
        ):
            line = _MARGIN_RE.sub(r"Margin: \1%", line)
        result_lines.append(line)
    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# KPI text normalization
# ---------------------------------------------------------------------------

# Known KPI label keywords, ordered longest-first so multi-word labels match
# before their sub-words (e.g. "retail area" before "area").
_KPI_LABELS: list[tuple[str, str]] = [
    # (canonical display name, regex fragment to match in source text)
    ("Store count",      r"store\s*count"),
    ("Stores",           r"stores?"),
    ("City Presence",    r"cit(?:y|ies)\s*presence"),
    ("Cities",           r"cit(?:y|ies)"),
    ("Retail Area",      r"retail\s*area"),
    ("Members",          r"members?"),
    ("Revenue",          r"revenue"),
    ("EBITDA",           r"ebitda"),
    ("PAT",              r"(?:pat|profit\s*after\s*tax)"),
    ("Gross Margin",     r"gross\s*margin"),
    ("EBITDA Margin",    r"ebitda\s*margin"),
    ("PAT Margin",       r"pat\s*margin"),
    ("Net Sales",        r"net\s*sales"),
    ("Same Store Sales", r"same\s*store\s*sales|sssg"),
    ("LTL Growth",       r"ltl\s*growth"),
    ("Sq ft",            r"sq\.?\s*ft\.?|square\s*feet"),
    ("Mn sq ft",         r"mn\s*sq\.?\s*ft\.?|million\s*sq"),
]

# Number pattern: integers with optional commas, decimals, and a unit suffix
# (Mn, Cr, Bn, Lakh, %, sq ft, etc.)
_NUMBER_PAT = r"[\d,]+(?:\.\d+)?\s*(?:Mn|Cr|Bn|Lakh|mn|cr|bn|lakh|%|sq\.?\s*ft\.?)?"

# Pre-compiled label patterns (longest-first order preserved from _KPI_LABELS)
_COMPILED_LABELS: list[tuple[str, re.Pattern]] = [
    (display, re.compile(pattern, re.IGNORECASE))
    for display, pattern in _KPI_LABELS
]


def _normalize_kpi_line(line: str) -> str:
    """
    Detect and rewrite a single line that contains one or more inline KPI
    pairs in the form  "<number> <label>"  or  "<label> <number>".

    Examples
    --------
    "1101 Store count 251 City Presence 14.7 Mn Retail Area"
        → "Store count: 1101. City Presence: 251. Retail Area: 14.7 Mn."

    "Revenue 4156 Cr  EBITDA 1163 Cr"
        → "Revenue: 4156 Cr. EBITDA: 1163 Cr."

    Lines that do not contain any recognised KPI label are returned unchanged.
    Lines that already contain "Label: value" patterns (e.g. already processed
    by _label_margin_vs_growth) are returned unchanged to avoid double-tagging.
    """
    # Quick exit: no digit → nothing to normalise
    if not re.search(r"\d", line):
        return line

    # Skip lines that are already in "Label: value" form — they were either
    # already normalised or tagged by _label_margin_vs_growth.
    if re.search(r"[A-Za-z]\s*:\s*[\d]", line):
        return line

    # Check whether any known label appears in the line
    matched_labels = [
        (display, pat)
        for display, pat in _COMPILED_LABELS
        if pat.search(line)
    ]
    if not matched_labels:
        return line

    # Find all number spans in the line.
    # Captures: integer/decimal with optional commas, followed by an optional
    # unit (Mn, Cr, Bn, Lakh) and/or a % sign.
    num_re = re.compile(
        r"(?<![:\d])"                                    # not preceded by colon or digit
        r"([\d,]+(?:\.\d+)?)"                            # the number itself
        r"(\s*%|"                                        # optional % …
        r"\s*(?:Mn|Cr|Bn|Lakh|mn|cr|bn|lakh)"           # … or unit word
        r"(?:\s+sq\.?\s*ft\.?)?"                         # … optionally followed by sq ft
        r")?",
        re.IGNORECASE,
    )
    numbers: list[tuple[int, int, str]] = []   # (start, end, text)
    for m in num_re.finditer(line):
        num_text = (m.group(1) + (m.group(2) or "")).strip()
        if not num_text:
            continue
        numbers.append((m.start(), m.start() + len(num_text), num_text))

    if not numbers:
        return line

    # Find all label spans for matched labels
    label_spans: list[tuple[int, int, str]] = []   # (start, end, display_name)
    for display, pat in matched_labels:
        for m in pat.finditer(line):
            label_spans.append((m.start(), m.end(), display))

    if not label_spans:
        return line

    # Sort both by position
    numbers.sort(key=lambda x: x[0])
    label_spans.sort(key=lambda x: x[0])

    # Deduplicate overlapping label spans: when two labels overlap (e.g.
    # "Stores" inside "Store count"), keep only the longest one.
    deduped_labels: list[tuple[int, int, str]] = []
    for ls, le, ldisplay in label_spans:
        dominated = any(
            kept_s <= ls and le <= kept_e
            for kept_s, kept_e, _ in deduped_labels
        )
        if dominated:
            continue
        deduped_labels = [
            (ks, ke, kd) for ks, ke, kd in deduped_labels
            if not (ls <= ks and ke <= le)
        ]
        deduped_labels.append((ls, le, ldisplay))

    label_spans = sorted(deduped_labels, key=lambda x: x[0])

    # Pair each label with its nearest number (before or after, within 60 chars)
    pairs: list[tuple[str, str]] = []   # (display_label, number_text)
    used_num_indices: set[int] = set()
    used_lbl_indices: set[int] = set()

    for li, (ls, le, ldisplay) in enumerate(label_spans):
        best_ni = None
        best_dist = float("inf")
        for ni, (ns, ne, ntext) in enumerate(numbers):
            if ni in used_num_indices:
                continue
            dist = min(abs(ls - ne), abs(ns - le))
            if dist < best_dist:
                best_dist = dist
                best_ni = ni
        if best_ni is not None and best_dist <= 60:
            pairs.append((ldisplay, numbers[best_ni][2]))
            used_num_indices.add(best_ni)
            used_lbl_indices.add(li)

    if not pairs:
        return line

    # Reconstruct as "Label: value." sentences
    normalized = "  ".join(f"{label}: {value}." for label, value in pairs)

    # Collect residual text (parts of the line not consumed by any pair)
    consumed_spans: list[tuple[int, int]] = []
    for li, (ls, le, _) in enumerate(label_spans):
        if li in used_lbl_indices:
            consumed_spans.append((ls, le))
    for ni, (ns, ne, _) in enumerate(numbers):
        if ni in used_num_indices:
            consumed_spans.append((ns, ne))

    consumed_spans.sort()
    residual_parts: list[str] = []
    prev = 0
    for cs, ce in consumed_spans:
        chunk = line[prev:cs].strip(" ,;:-")
        if chunk:
            residual_parts.append(chunk)
        prev = ce
    tail = line[prev:].strip(" ,;:-")
    if tail:
        residual_parts.append(tail)

    residual = "  ".join(p for p in residual_parts if p)
    if residual:
        return normalized + "  " + residual
    return normalized


def _normalize_kpi_text(text: str) -> str:
    """
    Apply KPI normalization line-by-line across a full page text.

    Lines that look like a run-on KPI block (multiple label-value pairs on
    one line, as pdfplumber sometimes produces from slide layouts) are
    rewritten into explicit key-value sentences.  All other lines pass through
    unchanged.
    """
    result: list[str] = []
    for line in text.splitlines():
        result.append(_normalize_kpi_line(line))
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Sentence-boundary splitting
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, keeping newlines as hard boundaries.
    Each line is treated as a sentence unit (appropriate for slide text).
    """
    sentences: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            # Further split on ". " within a line (prose paragraphs)
            parts = re.split(r"(?<=\.)\s+(?=[A-Z])", line)
            sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def _sentence_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    kpi_blocks: List[Tuple[int, int]],
    lines: List[str],
) -> List[str]:
    """
    Split text into chunks respecting:
    1. KPI blocks are atomic — never split across chunk boundaries.
    2. Splits happen at sentence/line boundaries, not mid-word.
    3. Non-KPI prose uses word-count overlap.
    """
    if not lines:
        return [text] if text.strip() else []

    # Build a set of line indices that belong to KPI blocks
    kpi_line_indices: set[int] = set()
    for start, end in kpi_blocks:
        for idx in range(start, end + 1):
            kpi_line_indices.add(idx)

    chunks: List[str] = []
    current_lines: List[str] = []
    current_words = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this line starts a KPI block
        block_start = next(
            (s for s, e in kpi_blocks if s == i), None
        )
        if block_start is not None:
            block_end = next(e for s, e in kpi_blocks if s == i)
            block_lines = lines[block_start : block_end + 1]
            block_text = "\n".join(block_lines)
            block_words = len(block_text.split())

            # If adding the block would overflow, flush current buffer first
            if current_words + block_words > chunk_size and current_lines:
                chunks.append("\n".join(current_lines))
                # Overlap: keep last `overlap` words worth of lines
                overlap_lines = _last_n_words_of_lines(current_lines, overlap)
                current_lines = overlap_lines
                current_words = sum(len(l.split()) for l in current_lines)

            # Always add the entire KPI block as one unit
            current_lines.extend(block_lines)
            current_words += block_words
            i = block_end + 1
            continue

        # Regular line
        line_words = len(line.split())

        # If a single line is itself longer than chunk_size, split it with
        # word-window fallback so we don't produce unbounded chunks.
        if line_words > chunk_size:
            # Flush current buffer first
            if current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_words = 0
            sub = _word_chunks(line, chunk_size, overlap)
            chunks.extend(sub[:-1])          # all but last go straight out
            current_lines = [sub[-1]]        # last sub-chunk continues
            current_words = len(sub[-1].split())
            i += 1
            continue

        if current_words + line_words > chunk_size and current_lines:
            chunks.append("\n".join(current_lines))
            overlap_lines = _last_n_words_of_lines(current_lines, overlap)
            current_lines = overlap_lines
            current_words = sum(len(l.split()) for l in current_lines)

        current_lines.append(line)
        current_words += line_words
        i += 1

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks if chunks else [text]


def _last_n_words_of_lines(lines: List[str], n_words: int) -> List[str]:
    """
    Return the tail of `lines` that contains approximately `n_words` words,
    keeping whole lines (never splitting a line mid-way).
    """
    result: List[str] = []
    count = 0
    for line in reversed(lines):
        w = len(line.split())
        if count + w > n_words and result:
            break
        result.insert(0, line)
        count += w
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _make_chunk_id(document_name: str, page_number: int, chunk_index: int) -> str:
    """Deterministic, stable chunk ID based on document + page + position."""
    raw = f"{document_name}::p{page_number}::c{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_chunks(
    pages: List[tuple],          # (page_number, section_title, text, ocr_used?)
    document_name: str,
    company_name: str = "",
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> List[Chunk]:
    """
    Convert parsed pages into Chunk objects ready for embedding and storage.

    Accepts pages as either 3-tuples (page_number, section_title, text) for
    backward compatibility, or 4-tuples (page_number, section_title, text,
    ocr_used) from the updated parser.
    """
    ingested_at = datetime.now(timezone.utc).isoformat()
    all_chunks: List[Chunk] = []

    for page_tuple in pages:
        # Support both 3-tuple (legacy) and 4-tuple (with ocr_used flag)
        if len(page_tuple) == 4:
            page_number, section_title, text, ocr_used = page_tuple
        else:
            page_number, section_title, text = page_tuple
            ocr_used = False

        # Pre-processing pass 1: label margin vs growth percentages first,
        # so "EBITDA margin 28%" becomes "EBITDA margin Margin: 28%" before
        # the KPI normalizer sees it — and the normalizer's already-labelled
        # guard then skips those lines cleanly.
        text = _label_margin_vs_growth(text)

        # Pre-processing pass 2: normalize inline KPI pairs into explicit
        # key-value sentences before embedding and storage.
        # Skips lines already in "Label: value" form.
        text_normalized = _normalize_kpi_text(text)
        kpi_normalized = text_normalized != text
        text = text_normalized

        # Detect KPI blocks in the (now normalized) page lines
        lines = [l for l in text.splitlines() if l.strip()]
        kpi_blocks = _detect_kpi_blocks(lines)

        # Split into sub-chunks respecting KPI block atomicity
        sub_texts = _sentence_chunks(text, chunk_size, chunk_overlap, kpi_blocks, lines)

        for idx, sub_text in enumerate(sub_texts):
            if not sub_text.strip():
                continue

            chunk_id = _make_chunk_id(document_name, page_number, idx)

            # Determine if this sub-chunk contains a KPI block
            sub_lines = [l for l in sub_text.splitlines() if l.strip()]
            sub_kpi_blocks = _detect_kpi_blocks(sub_lines)
            is_kpi_chunk = len(sub_kpi_blocks) > 0

            metadata: Dict[str, Any] = {
                "company": company_name,
                "document_name": document_name,
                "ingested_at": ingested_at,
                "chunk_index_on_page": idx,
                "total_chunks_on_page": len(sub_texts),
                "ocr_used": ocr_used,
                "is_kpi_chunk": is_kpi_chunk,
                "kpi_normalized": kpi_normalized,
            }

            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=sub_text,
                    page_number=page_number,
                    section_title=section_title,
                    metadata=metadata,
                )
            )

    return all_chunks
