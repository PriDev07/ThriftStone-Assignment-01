"""
Pydantic schemas for the Investor Presentation RAG API.

Covers internal data structures (Chunk) and all request/response bodies
for the three public endpoints: /health, /ingest, and /query.
"""
from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Internal chunk representation
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A single text unit produced by the ingestion pipeline."""

    chunk_id: str = Field(..., description="Stable, deterministic unique identifier.")
    text: str = Field(..., description="Cleaned, KPI-normalised text content.")
    page_number: int = Field(..., description="Source PDF page number (1-indexed).")
    section_title: str = Field(
        default="General",
        description="Canonical section tag derived from page content.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-chunk metadata: company, document, ingestion_timestamp, "
            "used_vision, kpi_normalized, chunk_index."
        ),
    )


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    """Response returned after a successful PDF ingestion."""

    status: str = Field(..., description="'success' on completion.")
    pages_processed: int = Field(..., description="Total pages extracted from the PDF.")
    chunks_created: int = Field(..., description="Total chunks stored in Qdrant.")
    vision_pages: int = Field(
        ...,
        description="Pages where the vision LLM fallback was invoked.",
    )
    message: str = Field(..., description="Human-readable summary of the ingestion run.")


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., min_length=3, description="Analyst-style question.")
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Number of chunks to retrieve from Qdrant.",
    )


class Citation(BaseModel):
    """A single source citation attached to an answer."""

    page: int = Field(..., description="PDF page number the evidence came from.")
    chunk_id: str = Field(..., description="Identifier of the source chunk.")
    excerpt: str = Field(
        ...,
        description="First 200 characters of the chunk text used as evidence.",
    )


class RetrievalInfo(BaseModel):
    """Metadata about the retrieval step."""

    top_k: int = Field(..., description="Number of chunks requested.")
    chunks_consulted: int = Field(..., description="Number of chunks actually returned.")


class QueryResponse(BaseModel):
    """Response returned by the /query endpoint."""

    answer: str = Field(..., description="Grounded answer with inline [Page X] citations.")
    citations: List[Citation] = Field(..., description="Structured source citations.")
    retrieval: RetrievalInfo = Field(..., description="Retrieval step metadata.")
    limitations: List[str] = Field(
        default_factory=list,
        description="Caveats or gaps noted when evidence is weak or absent.",
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response returned by the /health endpoint."""

    status: str = Field(..., description="'ok' when all systems are operational.")
    qdrant: str = Field(..., description="'connected' or 'disconnected'.")
    embedding_model: str = Field(..., description="Active sentence-transformers model name.")
    llm_model: str = Field(..., description="Active Groq LLM model name.")
    version: str = Field(..., description="API version string.")
