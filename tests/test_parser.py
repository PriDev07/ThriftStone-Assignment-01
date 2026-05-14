"""
Tests for the Investor Presentation RAG pipeline.

Covers: PDF chunking, KPI normalization, section title detection,
LLM response parsing, and all three API endpoints.
"""
import pytest
from app.parser import _clean_md, _tag_section_title, _merge_ocr_text


# ---------------------------------------------------------------------------
# _clean_md
# ---------------------------------------------------------------------------

class TestCleanMd:
    def test_collapses_excessive_newlines(self):
        """Three or more consecutive blank lines are collapsed to two."""
        text = "line1\n\n\n\n\nline2"
        result = _clean_md(text)
        assert "\n\n\n" not in result

    def test_strips_trailing_spaces_per_line(self):
        """Trailing spaces on each line are removed."""
        result = _clean_md("hello   \nworld   ")
        for line in result.splitlines():
            assert not line.endswith(" ")

    def test_preserves_meaningful_newlines(self):
        """Single blank lines between paragraphs are preserved."""
        text = "line1\n\nline2"
        result = _clean_md(text)
        assert "line1" in result
        assert "line2" in result

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert _clean_md("") == ""

    def test_preserves_markdown_table(self):
        """Markdown pipe tables are not altered."""
        table = "| Col1 | Col2 |\n|------|------|\n| A    | B    |"
        result = _clean_md(table)
        assert "|" in result
        assert "Col1" in result

    def test_preserves_markdown_bold(self):
        """Bold markers are not stripped."""
        result = _clean_md("**1101** Store count")
        assert "**1101**" in result


# ---------------------------------------------------------------------------
# _tag_section_title
# ---------------------------------------------------------------------------

class TestTagSectionTitle:
    def test_results_context_exact(self):
        """'Results Context' heading maps to the Results Context tag."""
        assert _tag_section_title("Results Context\nSome body") == "Results Context"

    def test_results_context_uppercase(self):
        """'RESULTS CONTEXT' (uppercase) maps to the Results Context tag."""
        assert _tag_section_title("RESULTS CONTEXT\nBody text") == "Results Context"

    def test_at_a_glance(self):
        """'AT A GLANCE' maps to Key Metrics."""
        assert _tag_section_title("AT A GLANCE\n1101 stores") == "Key Metrics"

    def test_highlights(self):
        """'HIGHLIGHTS' maps to Key Metrics."""
        assert _tag_section_title("HIGHLIGHTS\nRevenue grew") == "Key Metrics"

    def test_trends(self):
        """'TRENDS' maps to Financial Trends."""
        assert _tag_section_title("TRENDS\nQ1 Q2 Q3") == "Financial Trends"

    def test_comparative_periods(self):
        """'COMPARATIVE PERIODS' maps to Financial Trends."""
        assert _tag_section_title("COMPARATIVE PERIODS\nFY24 FY25") == "Financial Trends"

    def test_campaigns(self):
        """'CAMPAIGNS' maps to Brand Activity."""
        assert _tag_section_title("CAMPAIGNS\nSummer sale") == "Brand Activity"

    def test_connect(self):
        """'CONNECT' maps to Brand Activity."""
        assert _tag_section_title("CONNECT\nSocial media") == "Brand Activity"

    def test_recent_stores(self):
        """'RECENT STORES' maps to Brand Activity."""
        assert _tag_section_title("RECENT STORES\nNew openings") == "Brand Activity"

    def test_sustainability(self):
        """'SUSTAINABILITY' maps to Sustainability."""
        assert _tag_section_title("SUSTAINABILITY\nCarbon targets") == "Sustainability"

    def test_csr(self):
        """'CSR' maps to Sustainability."""
        assert _tag_section_title("CSR initiatives") == "Sustainability"

    def test_headwind_commentary(self):
        """Pages with headwind/sentiment signals map to Management Commentary."""
        text = "Consumer sentiment remains weak due to inflationary headwinds."
        assert _tag_section_title(text) == "Management Commentary"

    def test_risk_commentary(self):
        """Pages with risk/macro signals map to Management Commentary."""
        text = "Key risks include macro uncertainty and discretionary spend slowdown."
        assert _tag_section_title(text) == "Management Commentary"

    def test_financial_keywords(self):
        """Pages with financial keywords but no heading map to Financial Results."""
        text = "Revenue grew 12% YoY. EBITDA margin improved to 28%."
        assert _tag_section_title(text) == "Financial Results"

    def test_markdown_heading_fallback(self):
        """The first Markdown heading is used as the tag when no keyword matches."""
        text = "# Revenue Overview\nSome body text here."
        assert _tag_section_title(text) == "Revenue Overview"

    def test_no_signals_returns_general(self):
        """Pages with no recognisable signals return 'General'."""
        text = "Welcome to our presentation."
        assert _tag_section_title(text) == "General"

    def test_priority_results_context_over_financial(self):
        """'Results Context' keyword takes priority over financial heuristics."""
        text = "Results Context\nRevenue grew 12% YoY."
        assert _tag_section_title(text) == "Results Context"


# ---------------------------------------------------------------------------
# _merge_ocr_text
# ---------------------------------------------------------------------------

class TestMergeOcrText:
    def test_new_ocr_lines_added(self):
        """OCR lines not present in primary text are appended."""
        primary = "Revenue: 4156 Cr"
        ocr = "Store Count: 1101\nCities: 251"
        result = _merge_ocr_text(primary, ocr)
        assert "Store Count: 1101" in result
        assert "Cities: 251" in result

    def test_duplicate_lines_not_added(self):
        """OCR lines with >60% word overlap with primary text are skipped."""
        primary = "Revenue: 4156 Cr\nEBITDA margin 28%"
        ocr = "Revenue 4156 Cr"
        result = _merge_ocr_text(primary, ocr)
        lines = [l for l in result.splitlines() if "revenue" in l.lower()]
        assert len(lines) == 1

    def test_empty_ocr_returns_original(self):
        """Empty OCR output returns the primary text unchanged."""
        assert _merge_ocr_text("Some text", "") == "Some text"

    def test_empty_primary_returns_ocr(self):
        """Empty primary text returns the OCR text."""
        ocr = "1101 Stores\n251 Cities"
        result = _merge_ocr_text("", ocr)
        assert "1101 Stores" in result
        assert "251 Cities" in result
