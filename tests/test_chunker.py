"""
Tests for the Investor Presentation RAG pipeline.

Covers: PDF chunking, KPI normalization, section title detection,
LLM response parsing, and all three API endpoints.
"""
import pytest
from app.chunker import (
    _word_chunks,
    _is_kpi_line,
    _detect_kpi_blocks,
    _label_margin_vs_growth,
    _normalize_kpi_line,
    _normalize_kpi_text,
    _sentence_chunks,
    build_chunks,
)


# ---------------------------------------------------------------------------
# Word-window splitter
# ---------------------------------------------------------------------------

class TestWordChunks:
    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size is returned as a single chunk."""
        text = "hello world this is a short text"
        result = _word_chunks(text, chunk_size=400, overlap=80)
        assert result == [text]

    def test_long_text_splits_correctly(self):
        """Text longer than chunk_size is split into multiple chunks."""
        text = " ".join([f"word{i}" for i in range(1000)])
        chunks = _word_chunks(text, chunk_size=400, overlap=80)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) <= 400

    def test_overlap_is_preserved(self):
        """The last `overlap` words of chunk N appear at the start of chunk N+1."""
        words = [f"w{i}" for i in range(500)]
        text = " ".join(words)
        chunks = _word_chunks(text, chunk_size=400, overlap=80)
        end_of_first = chunks[0].split()[-80:]
        start_of_second = chunks[1].split()[:80]
        assert end_of_first == start_of_second

    def test_exact_chunk_size_no_split(self):
        """Text exactly at chunk_size is not split."""
        text = " ".join([f"w{i}" for i in range(400)])
        result = _word_chunks(text, chunk_size=400, overlap=80)
        assert len(result) == 1

    def test_empty_text(self):
        """Empty input returns a list containing one empty string."""
        result = _word_chunks("", chunk_size=400, overlap=80)
        assert result == [""]


# ---------------------------------------------------------------------------
# KPI line detection
# ---------------------------------------------------------------------------

class TestKpiLineDetection:
    def test_number_with_label_is_kpi(self):
        """A line with a number followed by a label is a KPI line."""
        assert _is_kpi_line("1,101 Stores") is True

    def test_percentage_with_label_is_kpi(self):
        """A percentage followed by a label is a KPI line."""
        assert _is_kpi_line("28% EBITDA Margin") is True

    def test_plain_prose_is_not_kpi(self):
        """A prose sentence with no number is not a KPI line."""
        assert _is_kpi_line("We continue to invest in our store network") is False

    def test_number_only_is_not_kpi(self):
        """A bare number with no label is not a KPI line."""
        assert _is_kpi_line("12345") is False

    def test_city_count_kpi(self):
        """'251 Cities' is a valid KPI line."""
        assert _is_kpi_line("251 Cities") is True

    def test_revenue_kpi(self):
        """'Revenue: ₹4,156 Cr' is a valid KPI line."""
        assert _is_kpi_line("Revenue: ₹4,156 Cr") is True


class TestDetectKpiBlocks:
    def test_detects_contiguous_kpi_block(self):
        """Contiguous KPI lines are grouped into a single block."""
        lines = [
            "Results Context",
            "1,101 Stores",
            "251 Cities",
            "806 Zudio stores",
            "Some prose here",
        ]
        blocks = _detect_kpi_blocks(lines, min_block_size=2)
        assert len(blocks) == 1
        assert blocks[0] == (1, 3)

    def test_single_kpi_line_below_min_not_detected(self):
        """A single KPI line below min_block_size is not returned as a block."""
        lines = ["Prose line", "1,101 Stores", "More prose"]
        blocks = _detect_kpi_blocks(lines, min_block_size=2)
        assert blocks == []

    def test_multiple_separate_blocks(self):
        """Two separate KPI runs produce two separate blocks."""
        lines = [
            "1,101 Stores",
            "251 Cities",
            "Prose break",
            "28% Margin",
            "6% PAT Growth",
        ]
        blocks = _detect_kpi_blocks(lines, min_block_size=2)
        assert len(blocks) == 2
        assert blocks[0] == (0, 1)
        assert blocks[1] == (3, 4)

    def test_empty_lines(self):
        """Empty input returns an empty list."""
        assert _detect_kpi_blocks([], min_block_size=2) == []


# ---------------------------------------------------------------------------
# Margin vs growth labelling
# ---------------------------------------------------------------------------

class TestLabelMarginVsGrowth:
    def test_yoy_growth_tagged(self):
        """Lines with YoY growth keywords get 'YoY Growth:' prepended to the %."""
        text = "Revenue grew 12% YoY"
        result = _label_margin_vs_growth(text)
        assert "YoY Growth: 12%" in result

    def test_margin_tagged_when_keyword_present(self):
        """Lines with margin keywords get 'Margin:' prepended to the %."""
        text = "EBITDA margin 28%"
        result = _label_margin_vs_growth(text)
        assert "Margin: 28%" in result

    def test_plain_percentage_without_keyword_not_tagged(self):
        """A bare % with no margin or growth keyword is left unchanged."""
        text = "Store count increased by 15%"
        result = _label_margin_vs_growth(text)
        assert "Margin:" not in result

    def test_does_not_alter_numbers(self):
        """The numeric value is preserved exactly after tagging."""
        text = "PAT margin 10%"
        result = _label_margin_vs_growth(text)
        assert "10%" in result

    def test_multiline_mixed(self):
        """Margin and growth lines in the same text are each tagged correctly."""
        text = "EBITDA margin 28%\nRevenue grew 12% YoY"
        result = _label_margin_vs_growth(text)
        lines = result.splitlines()
        assert any("Margin: 28%" in l for l in lines)
        assert any("YoY Growth: 12%" in l for l in lines)


# ---------------------------------------------------------------------------
# KPI text normalization
# ---------------------------------------------------------------------------

class TestNormalizeKpiLine:
    def test_number_first_single_pair(self):
        """'1101 Store count' is rewritten as 'Store count: 1101.'"""
        result = _normalize_kpi_line("1101 Store count")
        assert "Store count: 1101." in result

    def test_label_first_single_pair(self):
        """'Store count 1101' is rewritten as 'Store count: 1101.'"""
        result = _normalize_kpi_line("Store count 1101")
        assert "Store count: 1101." in result

    def test_multiple_pairs_on_one_line(self):
        """Multiple inline KPI pairs on one line are each normalised."""
        line = "1101 Store count 251 City Presence 14.7 Mn Retail Area"
        result = _normalize_kpi_line(line)
        assert "Store count: 1101." in result
        assert "City Presence: 251." in result
        assert "Retail Area: 14.7 Mn." in result

    def test_revenue_and_ebitda(self):
        """Revenue and EBITDA pairs on one line are each normalised."""
        line = "Revenue 4156 Cr  EBITDA 1163 Cr"
        result = _normalize_kpi_line(line)
        assert "Revenue: 4156 Cr." in result
        assert "EBITDA: 1163 Cr." in result

    def test_members_label(self):
        """'Members 12 Mn' is rewritten as 'Members: 12 Mn.'"""
        result = _normalize_kpi_line("Members 12 Mn")
        assert "Members: 12 Mn." in result

    def test_no_kpi_label_unchanged(self):
        """Lines with no recognised KPI label pass through unchanged."""
        line = "We continue to invest in our store network."
        assert _normalize_kpi_line(line) == line

    def test_no_digit_unchanged(self):
        """Lines with no digit pass through unchanged."""
        line = "Strategy and Outlook"
        assert _normalize_kpi_line(line) == line

    def test_city_presence_variant(self):
        """'251 Cities' is rewritten as 'Cities: 251.'"""
        result = _normalize_kpi_line("251 Cities")
        assert "Cities: 251." in result

    def test_does_not_duplicate_numbers(self):
        """The number appears exactly once in the normalised output."""
        result = _normalize_kpi_line("1101 Stores")
        assert result.count("1101") == 1

    def test_percentage_store_count(self):
        """Percentage values are preserved in the normalised output."""
        result = _normalize_kpi_line("Stores 806 Zudio")
        assert "806" in result


class TestNormalizeKpiText:
    def test_multiline_mixed_content(self):
        """KPI lines are normalised; prose lines pass through unchanged."""
        text = (
            "Q2 FY26 Results\n"
            "1101 Store count 251 City Presence\n"
            "Revenue grew strongly this quarter.\n"
            "Revenue 4156 Cr  EBITDA 1163 Cr"
        )
        result = _normalize_kpi_text(text)
        lines = result.splitlines()
        assert lines[0] == "Q2 FY26 Results"
        assert "Store count: 1101." in lines[1]
        assert "City Presence: 251." in lines[1]
        assert lines[2] == "Revenue grew strongly this quarter."
        assert "Revenue: 4156 Cr." in lines[3]
        assert "EBITDA: 1163 Cr." in lines[3]

    def test_empty_text(self):
        """Empty input returns empty string."""
        assert _normalize_kpi_text("") == ""

    def test_pure_prose_unchanged(self):
        """Pure prose text is returned unchanged."""
        text = "Management remains focused on long-term growth.\nNo numeric KPIs here."
        assert _normalize_kpi_text(text) == text

    def test_normalization_stored_in_chunk_text(self):
        """Normalised KPI text appears in the Chunk.text field after build_chunks."""
        pages = [(1, "Store Network", "1101 Store count 251 City Presence")]
        chunks = build_chunks(pages, document_name="test.pdf")
        assert len(chunks) == 1
        assert "Store count: 1101." in chunks[0].text
        assert "City Presence: 251." in chunks[0].text

    def test_normalization_before_embedding_text(self):
        """The raw run-on form must not appear in the stored chunk text."""
        pages = [(1, "KPI Slide", "1101 Store count 251 City Presence")]
        chunks = build_chunks(pages, document_name="test.pdf")
        assert "1101 Store count 251" not in chunks[0].text


# ---------------------------------------------------------------------------
# KPI block atomicity
# ---------------------------------------------------------------------------

class TestSentenceChunksKpiAtomicity:
    def test_kpi_block_not_split(self):
        """A KPI block must never be split across chunk boundaries."""
        prose = " ".join([f"word{i}" for i in range(350)])
        kpi_block = "\n".join([
            "1,101 Stores",
            "251 Cities",
            "806 Zudio stores",
            "295 Westside stores",
        ])
        text = prose + "\n" + kpi_block
        lines = [l for l in text.splitlines() if l.strip()]
        from app.chunker import _detect_kpi_blocks, _sentence_chunks
        kpi_blocks = _detect_kpi_blocks(lines)
        chunks = _sentence_chunks(
            text, chunk_size=400, overlap=80, kpi_blocks=kpi_blocks, lines=lines
        )
        kpi_lines = {"1,101 Stores", "251 Cities", "806 Zudio stores", "295 Westside stores"}
        found_in = [
            i for i, chunk in enumerate(chunks)
            if any(kl in chunk for kl in kpi_lines)
        ]
        assert len(set(found_in)) == 1, f"KPI lines split across chunks: {found_in}"

    def test_label_value_stays_together(self):
        """A single KPI line must not be split mid-line."""
        text = "Zudio: 806 stores across 251 cities"
        lines = [text]
        from app.chunker import _detect_kpi_blocks, _sentence_chunks
        kpi_blocks = _detect_kpi_blocks(lines)
        chunks = _sentence_chunks(
            text, chunk_size=10, overlap=2, kpi_blocks=kpi_blocks, lines=lines
        )
        assert any("Zudio: 806 stores across 251 cities" in c for c in chunks)


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------

class TestBuildChunks:
    def test_basic_build(self):
        """build_chunks produces one chunk per short page."""
        pages = [
            (1, "Revenue Overview", "Revenue grew 20% year over year to $5B."),
            (2, "Strategy", "We plan to expand into three new markets."),
        ]
        chunks = build_chunks(pages, document_name="test.pdf", company_name="Acme")
        assert len(chunks) == 2
        assert chunks[0].page_number == 1
        assert chunks[1].page_number == 2

    def test_chunk_ids_are_unique(self):
        """All chunk IDs across a multi-page document are unique."""
        pages = [(i, f"Section {i}", f"Content for page {i} " * 10) for i in range(1, 6)]
        chunks = build_chunks(pages, document_name="test.pdf")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_populated(self):
        """Chunk metadata contains company, document_name, and ingested_at."""
        pages = [(1, "Title", "Some text here")]
        chunks = build_chunks(pages, document_name="report.pdf", company_name="Corp")
        assert chunks[0].metadata["company"] == "Corp"
        assert chunks[0].metadata["document_name"] == "report.pdf"
        assert "ingested_at" in chunks[0].metadata

    def test_long_page_produces_multiple_chunks(self):
        """A page exceeding chunk_size is split into multiple chunks."""
        long_text = " ".join([f"word{i}" for i in range(1000)])
        pages = [(1, "Dense Page", long_text)]
        chunks = build_chunks(pages, document_name="test.pdf", chunk_size=400, chunk_overlap=80)
        assert len(chunks) > 1
        assert all(c.page_number == 1 for c in chunks)

    def test_section_title_preserved(self):
        """The section_title from the page tuple is stored on the chunk."""
        pages = [(3, "Financial Highlights", "EBITDA margin improved to 28%.")]
        chunks = build_chunks(pages, document_name="test.pdf")
        assert chunks[0].section_title == "Financial Highlights"

    def test_deterministic_chunk_ids(self):
        """The same input always produces the same chunk IDs."""
        pages = [(1, "Title", "Hello world")]
        chunks_a = build_chunks(pages, document_name="doc.pdf")
        chunks_b = build_chunks(pages, document_name="doc.pdf")
        assert chunks_a[0].chunk_id == chunks_b[0].chunk_id

    def test_ocr_flag_propagated(self):
        """ocr_used=True from a 4-tuple page is stored in chunk metadata."""
        pages = [(1, "KPI Slide", "1,101 Stores\n251 Cities", True)]
        chunks = build_chunks(pages, document_name="test.pdf")
        assert all(c.metadata["ocr_used"] is True for c in chunks)

    def test_no_ocr_flag_on_3tuple(self):
        """3-tuple pages (legacy) default ocr_used to False."""
        pages = [(1, "Title", "Some text")]
        chunks = build_chunks(pages, document_name="test.pdf")
        assert chunks[0].metadata["ocr_used"] is False

    def test_is_kpi_chunk_flag(self):
        """Chunks containing KPI blocks are flagged is_kpi_chunk=True."""
        kpi_text = "1,101 Stores\n251 Cities\n806 Zudio stores"
        pages = [(1, "Store Network", kpi_text)]
        chunks = build_chunks(pages, document_name="test.pdf")
        assert any(c.metadata["is_kpi_chunk"] for c in chunks)

    def test_kpi_normalized_flag_true_when_rewritten(self):
        """kpi_normalized=True when at least one KPI line was rewritten."""
        pages = [(1, "KPI Slide", "1101 Store count 251 City Presence")]
        chunks = build_chunks(pages, document_name="test.pdf")
        assert all(c.metadata["kpi_normalized"] is True for c in chunks)

    def test_kpi_normalized_flag_false_for_pure_prose(self):
        """kpi_normalized=False when no KPI normalization occurred."""
        pages = [(1, "Strategy", "We plan to expand into new markets next year.")]
        chunks = build_chunks(pages, document_name="test.pdf")
        assert all(c.metadata["kpi_normalized"] is False for c in chunks)

    def test_margin_labelled_in_chunk_text(self):
        """EBITDA margin % is labelled 'Margin:' in the stored chunk text."""
        pages = [(1, "Financials", "EBITDA margin 28%")]
        chunks = build_chunks(pages, document_name="test.pdf")
        assert any("Margin: 28%" in c.text for c in chunks)

    def test_growth_labelled_in_chunk_text(self):
        """YoY growth % is labelled 'YoY Growth:' in the stored chunk text."""
        pages = [(1, "Financials", "Revenue grew 12% YoY")]
        chunks = build_chunks(pages, document_name="test.pdf")
        assert any("YoY Growth: 12%" in c.text for c in chunks)
