"""
Tests for the Investor Presentation RAG pipeline.

Covers: PDF chunking, KPI normalization, section title detection,
LLM response parsing, and all three API endpoints.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.embedder import embed_query, embed_texts


class TestEmbedder:
    def test_embed_query_returns_list_of_floats(self):
        """embed_query returns a flat list of Python floats."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.5] * 384)

        with patch("app.embedder._get_model", return_value=mock_model):
            result = embed_query("What was revenue in Q2 FY26?")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)
        assert len(result) == 384

    def test_embed_query_passes_string_directly(self):
        """embed_query passes the question string directly without any prefix."""
        captured: list[str] = []

        def fake_encode(text, **kwargs):
            captured.append(text)
            return np.array([0.1] * 384)

        mock_model = MagicMock()
        mock_model.encode.side_effect = fake_encode

        with patch("app.embedder._get_model", return_value=mock_model):
            embed_query("What was revenue in Q2 FY26?")

        assert len(captured) == 1
        assert captured[0] == "What was revenue in Q2 FY26?"

    def test_embed_texts_does_not_modify_input(self):
        """embed_texts passes document text through unchanged."""
        captured: list = []

        def fake_encode(texts, **kwargs):
            captured.extend(texts)
            return np.array([[0.1] * 384, [0.2] * 384])

        mock_model = MagicMock()
        mock_model.encode.side_effect = fake_encode

        docs = ["Store count: 1101.", "Revenue: ₹4,724 Cr."]
        with patch("app.embedder._get_model", return_value=mock_model):
            embed_texts(docs)

        assert captured == docs

    def test_embed_texts_returns_list_of_lists(self):
        """embed_texts returns a list of float lists, one per input text."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])

        with patch("app.embedder._get_model", return_value=mock_model):
            result = embed_texts(["doc one", "doc two"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)
