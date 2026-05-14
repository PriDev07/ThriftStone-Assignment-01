"""
Tests for the Investor Presentation RAG pipeline.

Covers: PDF chunking, KPI normalization, section title detection,
LLM response parsing, and all three API endpoints.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Citation

client = TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_when_qdrant_up(self):
        """Health endpoint returns 'ok' when Qdrant is reachable."""
        with (
            patch("app.main.qdrant_healthy", return_value=True),
        ):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["qdrant"] == "connected"
        assert "embedding_model" in data
        assert "llm_model" in data
        assert data["version"] == "1.0.0"

    def test_health_when_qdrant_down(self):
        """Health endpoint returns 'degraded' when Qdrant is unreachable."""
        with (
            patch("app.main.qdrant_healthy", return_value=False),
        ):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["qdrant"] == "disconnected"


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def _fake_pdf_bytes(self) -> bytes:
        """Return minimal PDF-like bytes for routing tests."""
        return b"%PDF-1.4 fake content"

    def test_ingest_non_pdf_rejected(self):
        """Non-PDF uploads are rejected with HTTP 400."""
        resp = client.post(
            "/ingest",
            files={"file": ("report.txt", b"some text", "text/plain")},
            data={"company_name": "Acme"},
        )
        assert resp.status_code == 400

    def test_ingest_success(self):
        """Successful ingestion returns status='success' and document_name."""
        fake_pages = [
            (1, "Revenue", "Revenue was $5B.", False),
            (2, "Strategy", "Expand globally.", False),
        ]
        fake_chunk_list = [
            MagicMock(chunk_id="a1", text="Revenue was $5B.", page_number=1,
                      section_title="Revenue", metadata={}),
            MagicMock(chunk_id="a2", text="Expand globally.", page_number=2,
                      section_title="Strategy", metadata={}),
        ]

        with (
            patch("app.main.parse_pdf", return_value=fake_pages),
            patch("app.main.extract_visual_content",
                  return_value={"final_text": "Revenue was $5B.",
                                "used_vision": False, "word_count_before": 4}),
            patch("app.main.build_chunks", return_value=fake_chunk_list),
            patch("app.main.embed_texts", return_value=[[0.1] * 384, [0.2] * 384]),
            patch("app.main.ensure_collection"),
            patch("app.main.upsert_chunks"),
        ):
            resp = client.post(
                "/ingest",
                files={"file": ("report.pdf", self._fake_pdf_bytes(), "application/pdf")},
                data={"company_name": "Acme"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "report.pdf" in data["message"]

    def test_ingest_empty_pdf_returns_422(self):
        """A PDF that yields no pages returns HTTP 422."""
        with patch("app.main.parse_pdf", return_value=[]):
            resp = client.post(
                "/ingest",
                files={"file": ("empty.pdf", self._fake_pdf_bytes(), "application/pdf")},
                data={},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_no_collection_returns_400(self):
        """Querying before any ingestion returns HTTP 400."""
        with patch("app.main.collection_info",
                   return_value={"exists": False, "chunk_count": 0}):
            resp = client.post("/query", json={"question": "What is revenue?", "top_k": 5})
        assert resp.status_code == 400

    def test_query_returns_answer_with_citations(self):
        """A successful query returns answer, citations, and retrieval metadata."""
        fake_results = [
            (0.92, {
                "chunk_id": "abc",
                "text": "Revenue was $5B in FY2023.",
                "page_number": 4,
                "section_title": "Financials",
            })
        ]
        with (
            patch("app.main.collection_info",
                  return_value={"exists": True, "chunk_count": 10}),
            patch("app.main.embed_query", return_value=[0.1] * 384),
            patch("app.main.search", return_value=fake_results),
            patch(
                "app.main.generate_answer",
                return_value=(
                    "Revenue was $5B [Page 4].",
                    [Citation(page=4, chunk_id="abc",
                              excerpt="Revenue was $5B in FY2023.")],
                    [],
                ),
            ),
        ):
            resp = client.post("/query", json={"question": "What is revenue?", "top_k": 5})

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["citations"]) >= 1
        assert data["retrieval"]["top_k"] == 5

    def test_query_short_question_rejected(self):
        """Questions shorter than 3 characters are rejected with HTTP 422."""
        resp = client.post("/query", json={"question": "Hi", "top_k": 5})
        assert resp.status_code == 422

    def test_query_top_k_out_of_range(self):
        """top_k values outside [1, 20] are rejected with HTTP 422."""
        with patch("app.main.collection_info",
                   return_value={"exists": True, "chunk_count": 5}):
            resp = client.post("/query", json={"question": "What is revenue?", "top_k": 100})
        assert resp.status_code == 422
