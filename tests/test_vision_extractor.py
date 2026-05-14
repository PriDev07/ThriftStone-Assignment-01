"""
Tests for the Investor Presentation RAG pipeline.

Covers: PDF chunking, KPI normalization, section title detection,
LLM response parsing, and all three API endpoints.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.vision_extractor import (
    SPARSE_PAGE_MIN_WORDS,
    VISION_MODEL,
    describe_page_with_vision,
    extract_page_as_base64,
    extract_visual_content,
    is_content_sparse,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_sparse_page_min_words(self):
        """SPARSE_PAGE_MIN_WORDS is set to 30."""
        assert SPARSE_PAGE_MIN_WORDS == 30

    def test_vision_model(self):
        """VISION_MODEL points to the correct llama-4-scout model."""
        assert VISION_MODEL == "meta-llama/llama-4-scout-17b-16e-instruct"


# ---------------------------------------------------------------------------
# is_content_sparse
# ---------------------------------------------------------------------------

class TestIsContentSparse:
    def test_empty_text_is_sparse(self):
        """Empty text is always sparse."""
        assert is_content_sparse("") is True

    def test_below_30_words_is_sparse(self):
        """Fewer than 30 words triggers the word-count condition."""
        assert is_content_sparse("hello world this is short") is True

    def test_exactly_30_words_is_not_sparse(self):
        """Exactly 30 words does not trigger the word-count condition."""
        text = " ".join(["word"] * 30)
        assert is_content_sparse(text) is False

    def test_above_30_words_is_not_sparse(self):
        """More than 30 words does not trigger the word-count condition."""
        text = " ".join(["word"] * 50)
        assert is_content_sparse(text) is False

    def test_placeholder_lines_stripped_before_count(self):
        """pymupdf4llm placeholder lines are stripped before counting words."""
        text = "\n".join([
            "==> image_001.png <==",
            "----- figure -----",
            "==> chart.png <==",
        ])
        assert is_content_sparse(text) is True

    def test_placeholder_plus_few_words_still_sparse(self):
        """Placeholder lines plus a few real words still counts as sparse."""
        text = "==> image.png <==\nRevenue grew"
        assert is_content_sparse(text) is True

    def test_chart_indicator_without_large_numbers_is_sparse(self):
        """Chart indicator present but no 3-digit number triggers sparsity."""
        text = (
            "Store Count\n"
            "Q2 FY22 Q2 FY23 Q2 FY24 Q2 FY25 Q2 FY26\n"
            "19 stores opened and 6 consolidated in Q2FY26\n"
            "6 Mn+ Retail area\n"
            "88 City presence\n"
        )
        assert is_content_sparse(text) is True

    def test_chart_indicator_with_large_numbers_is_not_sparse(self):
        """Chart indicator plus 3-digit numbers and enough words is not sparse."""
        text = (
            "Store Count\n"
            "Q2 FY22: 261 stores\n"
            "Q2 FY23: 350 stores\n"
            "Q2 FY24: 450 stores\n"
            "Q2 FY25: 600 stores\n"
            "Q2 FY26: 806 stores\n"
            "Total store network has grown significantly over the past four years "
            "driven by Zudio expansion into tier-2 and tier-3 cities across India.\n"
        )
        assert is_content_sparse(text) is False

    def test_revenue_indicator_without_large_numbers_is_sparse(self):
        """Revenue keyword plus filler words but no 3-digit number is sparse."""
        filler = " ".join(["word"] * 25)
        text = "Revenue\nQ2 FY22 Q2 FY23 Q2 FY24 Q2 FY25 Q2 FY26\nGrowth trend\n" + filler
        assert is_content_sparse(text) is True

    def test_normal_prose_not_sparse(self):
        """Dense prose with no chart indicators is not sparse."""
        text = (
            "Management remains focused on long-term growth across all segments. "
            "The company plans to open 150 new stores in tier-2 and tier-3 cities "
            "over the next 24 months, with a focus on the Zudio brand which has "
            "shown strong consumer traction in value fashion."
        )
        assert is_content_sparse(text) is False

    def test_dotall_placeholder_removal(self):
        """Multi-line placeholder blocks are stripped correctly."""
        text = "==> some\nlong\nimage\nblock <==\nRevenue grew"
        assert is_content_sparse(text) is True


# ---------------------------------------------------------------------------
# extract_page_as_base64
# ---------------------------------------------------------------------------

class TestExtractPageAsBase64:
    def test_returns_empty_on_missing_pdf(self):
        """A non-existent PDF path returns an empty string."""
        result = extract_page_as_base64("/nonexistent/path.pdf", 1)
        assert result == ""

    def test_returns_base64_string_on_success(self):
        """A valid page render returns a non-empty base64 string."""
        import base64
        import io
        from PIL import Image as PILImage

        fake_img = PILImage.new("RGB", (100, 100), color="white")
        buf = io.BytesIO()
        fake_img.save(buf, format="PNG")
        buf.seek(0)
        expected_b64 = base64.b64encode(buf.read()).decode("utf-8")

        with (
            patch("app.vision_extractor._PDF2IMAGE_AVAILABLE", True),
            patch("app.vision_extractor._convert_from_path", return_value=[fake_img]),
        ):
            result = extract_page_as_base64("fake.pdf", 3)

        assert result == expected_b64

    def test_returns_empty_when_pdf2image_not_available(self):
        """Returns empty string when pdf2image is not installed."""
        with patch("app.vision_extractor._PDF2IMAGE_AVAILABLE", False):
            result = extract_page_as_base64("fake.pdf", 1)
        assert result == ""


# ---------------------------------------------------------------------------
# describe_page_with_vision
# ---------------------------------------------------------------------------

class TestDescribePageWithVision:
    def test_returns_empty_on_empty_input(self):
        """Empty base64 input returns empty string without calling the API."""
        assert describe_page_with_vision("") == ""

    def test_returns_vision_response_on_success(self):
        """A successful API call returns the model's structured description."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "SLIDE_TOPIC: Store count trend\n"
            "KEY_METRICS:\n- Store Count Q2FY26: 806"
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("app.vision_extractor._get_client", return_value=mock_client):
            result = describe_page_with_vision("fakeb64==")

        assert "Store Count" in result
        assert "806" in result

    def test_returns_empty_on_api_failure(self):
        """An API exception returns empty string without propagating."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        with patch("app.vision_extractor._get_client", return_value=mock_client):
            result = describe_page_with_vision("fakeb64==")

        assert result == ""

    def test_image_sent_as_data_url(self):
        """The image is sent as a data:image/png;base64,... URL."""
        captured_messages = []
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "SLIDE_TOPIC: Test"
        mock_client = MagicMock()

        def capture_call(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return mock_response

        mock_client.chat.completions.create.side_effect = capture_call

        with patch("app.vision_extractor._get_client", return_value=mock_client):
            describe_page_with_vision("abc123")

        assert len(captured_messages) == 1
        content = captured_messages[0]["content"]
        image_part = next(p for p in content if p["type"] == "image_url")
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
        assert "abc123" in image_part["image_url"]["url"]


# ---------------------------------------------------------------------------
# extract_visual_content
# ---------------------------------------------------------------------------

class TestExtractVisualContent:
    def _dense_text(self) -> str:
        """Return a 50-word text that is not sparse."""
        return " ".join(["word"] * 50)

    def test_returns_correct_keys(self):
        """Return dict always has exactly the three required keys."""
        result = extract_visual_content("fake.pdf", 1, self._dense_text())
        assert set(result.keys()) == {"final_text", "used_vision", "word_count_before"}

    def test_skips_vision_when_not_sparse(self):
        """Dense text bypasses vision extraction entirely."""
        result = extract_visual_content("fake.pdf", 1, self._dense_text())
        assert result["used_vision"] is False
        assert result["final_text"] == self._dense_text()
        assert result["word_count_before"] == 50

    def test_skips_vision_when_base64_empty(self):
        """Vision is skipped when page rendering fails."""
        with patch("app.vision_extractor.extract_page_as_base64", return_value=""):
            result = extract_visual_content("fake.pdf", 1, "few words")
        assert result["used_vision"] is False

    def test_skips_vision_when_description_empty(self):
        """Vision is skipped when the API returns an empty description."""
        with (
            patch("app.vision_extractor.extract_page_as_base64", return_value="b64data"),
            patch("app.vision_extractor.describe_page_with_vision", return_value=""),
        ):
            result = extract_visual_content("fake.pdf", 1, "few words")
        assert result["used_vision"] is False

    def test_merges_vision_and_existing_text(self):
        """Vision description is prepended to existing text with a separator."""
        vision_desc = "SLIDE_TOPIC: Store count\nKEY_METRICS:\n- Store Count: 806"
        existing = "Some existing text"
        with (
            patch("app.vision_extractor.extract_page_as_base64", return_value="b64"),
            patch("app.vision_extractor.describe_page_with_vision", return_value=vision_desc),
        ):
            result = extract_visual_content("fake.pdf", 1, existing)

        assert result["used_vision"] is True
        assert vision_desc in result["final_text"]
        assert "--- TEXT EXTRACTION ---" in result["final_text"]
        assert existing in result["final_text"]
        assert result["final_text"].index(vision_desc) < result["final_text"].index(existing)

    def test_vision_only_when_no_existing_text(self):
        """When existing text is empty, final_text is just the vision description."""
        vision_desc = "SLIDE_TOPIC: Revenue chart"
        with (
            patch("app.vision_extractor.extract_page_as_base64", return_value="b64"),
            patch("app.vision_extractor.describe_page_with_vision", return_value=vision_desc),
        ):
            result = extract_visual_content("fake.pdf", 1, "")

        assert result["used_vision"] is True
        assert result["final_text"] == vision_desc
        assert "--- TEXT EXTRACTION ---" not in result["final_text"]

    def test_word_count_before_reflects_original(self):
        """word_count_before reflects the original text, not the merged result."""
        original = "five words here now"
        with (
            patch("app.vision_extractor.extract_page_as_base64", return_value="b64"),
            patch("app.vision_extractor.describe_page_with_vision", return_value="Vision text"),
        ):
            result = extract_visual_content("fake.pdf", 1, original)
        assert result["word_count_before"] == 4

    def test_no_exception_raised_on_any_failure(self):
        """All internal failures are caught; no exception propagates to the caller."""
        with patch(
            "app.vision_extractor.extract_page_as_base64",
            side_effect=RuntimeError("unexpected"),
        ):
            result = extract_visual_content("fake.pdf", 1, "few words")
        assert result["used_vision"] is False
