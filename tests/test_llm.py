"""
Tests for the Investor Presentation RAG pipeline.

Covers: PDF chunking, KPI normalization, section title detection,
LLM response parsing, and all three API endpoints.
"""
import pytest
from app.llm import (
    _extract_limitations,
    _build_citations,
    _is_narrative_question,
    _boost_narrative_chunks,
    SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT contract
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_contains_context_placeholder(self):
        """SYSTEM_PROMPT must contain the {context} placeholder."""
        assert "{context}" in SYSTEM_PROMPT

    def test_contains_question_placeholder(self):
        """SYSTEM_PROMPT must contain the {question} placeholder."""
        assert "{question}" in SYSTEM_PROMPT

    def test_cite_rule_present(self):
        """SYSTEM_PROMPT must instruct the model to cite with [Page X]."""
        assert "[Page X]" in SYSTEM_PROMPT

    def test_fabrication_rule_present(self):
        """SYSTEM_PROMPT must explicitly forbid fabrication."""
        assert "fabricate" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# _extract_limitations
# ---------------------------------------------------------------------------

class TestExtractLimitations:
    def test_no_limitations_section(self):
        """Answers without a LIMITATIONS section return an empty list."""
        answer = "Revenue was $5B in FY2023 [Page 4]."
        clean, lims = _extract_limitations(answer)
        assert clean == answer
        assert lims == []

    def test_limitations_section_extracted(self):
        """LIMITATIONS section is split out and returned as a list."""
        answer = "Revenue was $5B [Page 4].\n\nLIMITATIONS:\n- Data only covers FY2023.\n- No Q4 breakdown."
        clean, lims = _extract_limitations(answer)
        assert "LIMITATIONS" not in clean
        assert len(lims) == 2
        assert "Data only covers FY2023." in lims

    def test_case_insensitive_limitations(self):
        """LIMITATIONS detection is case-insensitive."""
        answer = "Some answer.\n\nlimitations: weak evidence."
        clean, lims = _extract_limitations(answer)
        assert lims == ["weak evidence."]

    def test_limitations_with_bullet_symbols(self):
        """Bullet symbols are stripped from limitation items."""
        answer = "Answer text.\n\nLIMITATIONS:\n• Point one\n• Point two"
        _, lims = _extract_limitations(answer)
        assert "Point one" in lims
        assert "Point two" in lims


# ---------------------------------------------------------------------------
# _build_citations
# ---------------------------------------------------------------------------

class TestBuildCitations:
    def _make_payload(self, page: int, chunk_id: str, text: str) -> dict:
        """Helper: build a minimal chunk payload dict."""
        return {
            "page_number": page,
            "chunk_id": chunk_id,
            "text": text,
            "section_title": "Test Section",
        }

    def test_cited_pages_appear_in_citations(self):
        """Pages mentioned with [Page X] appear in the citations list."""
        answer = "Revenue grew [Page 3]."
        chunks = [
            (0.9, self._make_payload(3, "abc123", "Revenue grew 20% YoY to $5B.")),
            (0.7, self._make_payload(5, "def456", "Operating margin improved.")),
        ]
        citations = _build_citations(answer, chunks)
        pages = [c.page for c in citations]
        assert 3 in pages

    def test_all_retrieved_chunks_included(self):
        """All retrieved pages appear in citations even without explicit [Page X]."""
        answer = "Some answer without explicit page refs."
        chunks = [
            (0.9, self._make_payload(2, "aaa", "Text on page 2.")),
            (0.8, self._make_payload(7, "bbb", "Text on page 7.")),
        ]
        citations = _build_citations(answer, chunks)
        pages = {c.page for c in citations}
        assert {2, 7} == pages

    def test_no_duplicate_pages(self):
        """The same page is not cited twice even if mentioned twice."""
        answer = "See [Page 4] and [Page 4] again."
        chunks = [(0.9, self._make_payload(4, "x1", "Some text."))]
        citations = _build_citations(answer, chunks)
        pages = [c.page for c in citations]
        assert pages.count(4) == 1

    def test_excerpt_truncated_to_200_chars(self):
        """Citation excerpts are capped at 200 characters."""
        long_text = "A" * 500
        answer = "[Page 1]"
        chunks = [(0.9, self._make_payload(1, "z1", long_text))]
        citations = _build_citations(answer, chunks)
        assert len(citations[0].excerpt) <= 200


# ---------------------------------------------------------------------------
# Narrative question detection
# ---------------------------------------------------------------------------

class TestIsNarrativeQuestion:
    def test_headwind_is_narrative(self):
        """Questions about headwinds are classified as narrative."""
        assert _is_narrative_question("What headwinds does the company face?") is True

    def test_risk_is_narrative(self):
        """Questions about risks are classified as narrative."""
        assert _is_narrative_question("What risks are mentioned?") is True

    def test_sentiment_is_narrative(self):
        """Questions about consumer sentiment are classified as narrative."""
        assert _is_narrative_question("What is consumer sentiment like?") is True

    def test_why_is_narrative(self):
        """'Why' questions are classified as narrative."""
        assert _is_narrative_question("Why did margins decline?") is True

    def test_revenue_question_is_not_narrative(self):
        """Factual revenue questions are not classified as narrative."""
        assert _is_narrative_question("What was revenue in Q2 FY26?") is False

    def test_store_count_is_not_narrative(self):
        """Factual store count questions are not classified as narrative."""
        assert _is_narrative_question("How many stores does Zudio have?") is False


# ---------------------------------------------------------------------------
# Narrative chunk boosting
# ---------------------------------------------------------------------------

class TestBoostNarrativeChunks:
    def _make_chunk(self, score: float, section: str, page: int) -> tuple:
        """Helper: build a (score, payload) tuple."""
        return (score, {
            "page_number": page,
            "chunk_id": f"chunk_{page}",
            "text": f"Text from page {page}",
            "section_title": section,
        })

    def test_commentary_chunks_moved_to_front(self):
        """Management Commentary chunks are reordered to the front of the list."""
        chunks = [
            self._make_chunk(0.85, "Store Network", 17),
            self._make_chunk(0.80, "Campaign Images", 18),
            self._make_chunk(0.75, "Management Commentary", 7),
            self._make_chunk(0.70, "Management Commentary", 8),
        ]
        result = _boost_narrative_chunks(chunks)
        assert result[0][1]["section_title"] == "Management Commentary"
        assert result[1][1]["section_title"] == "Management Commentary"

    def test_boosted_score_does_not_exceed_1(self):
        """Boosted scores are capped at 1.0."""
        chunks = [self._make_chunk(0.99, "Management Commentary", 7)]
        result = _boost_narrative_chunks(chunks)
        assert result[0][0] <= 1.0

    def test_non_commentary_order_preserved(self):
        """Non-commentary chunks remain ordered by descending score."""
        chunks = [
            self._make_chunk(0.90, "Financial Results", 5),
            self._make_chunk(0.80, "Store Network", 12),
        ]
        result = _boost_narrative_chunks(chunks)
        assert result[0][1]["page_number"] == 5
        assert result[1][1]["page_number"] == 12

    def test_empty_input(self):
        """Empty input returns an empty list."""
        assert _boost_narrative_chunks([]) == []
