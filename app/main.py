"""
Investor Presentation RAG — FastAPI application.

Exposes three endpoints: health check, PDF ingestion, and question
answering with page-level citations.
"""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.chunker import build_chunks
from app.config import settings
from app.embedder import embed_query, embed_texts
from app.llm import generate_answer
from app.models import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievalInfo,
)
from app.parser import parse_pdf
from app.vector_store import (
    collection_info,
    ensure_collection,
    qdrant_healthy,
    search,
    upsert_chunks,
)
from app.vision_extractor import extract_visual_content

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Investor Presentation RAG",
    description=(
        "Ingest an investor presentation PDF and answer analyst-style "
        "questions with page-level citations."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Return service liveness status and key configuration values."""
    connected = qdrant_healthy()
    return HealthResponse(
        status="ok" if connected else "degraded",
        qdrant="connected" if connected else "disconnected",
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(
    file: UploadFile = File(..., description="Investor presentation PDF"),
    company_name: str = Form(default="", description="Optional company name tag"),
    force: bool = Query(
        default=False,
        description=(
            "Drop and recreate the Qdrant collection before ingesting. "
            "Use this to guarantee no stale chunks from a previous run persist."
        ),
    ),
) -> IngestResponse:
    """
    Ingest a PDF investor presentation into the vector store.

    Pipeline:
      1. Parse pages with pymupdf4llm (structured Markdown per page).
      2. For image-heavy pages, call the vision LLM to extract chart data.
      3. Normalise KPI label-value pairs and label margin vs growth percentages.
      4. Split into sentence-boundary chunks, keeping KPI blocks atomic.
      5. Embed with BAAI/bge-small-en-v1.5.
      6. Upsert into Qdrant.

    Pass ``?force=true`` to wipe the collection before ingesting.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # --- Parse ---
        try:
            raw_pages = parse_pdf(tmp_path)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"PDF parsing failed: {exc}",
            )

        if not raw_pages:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No text could be extracted from the PDF. "
                    "Ensure it is not a scanned or image-only document."
                ),
            )

        document_name = file.filename
        logger.info("Ingesting %s — %d pages found", document_name, len(raw_pages))

        # --- Vision augmentation for sparse pages ---
        augmented_pages: list[tuple] = []
        vision_pages = 0

        for page_number, section_title, text, ocr_used in raw_pages:
            vision_result = extract_visual_content(
                pdf_path=str(tmp_path),
                page_number=page_number,
                existing_text=text,
            )
            final_text = vision_result["final_text"]
            used_vision = vision_result["used_vision"]
            if used_vision:
                vision_pages += 1

            logger.info(
                "Page %d: %d words extracted, vision=%s, section='%s'",
                page_number,
                vision_result["word_count_before"],
                used_vision,
                section_title,
            )
            augmented_pages.append(
                (page_number, section_title, final_text, ocr_used or used_vision)
            )

        # --- Chunk ---
        chunks = build_chunks(
            pages=augmented_pages,
            document_name=document_name,
            company_name=company_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # Stamp used_vision into each chunk's metadata
        page_vision_map = {p[0]: p[3] for p in augmented_pages}
        for chunk in chunks:
            chunk.metadata["used_vision"] = page_vision_map.get(
                chunk.page_number, False
            )

        # --- Embed ---
        vectors = embed_texts([c.text for c in chunks])

        # --- Store ---
        ensure_collection(recreate=force)
        upsert_chunks(chunks, vectors)

        logger.info(
            "Ingestion complete: %d chunks stored (%d pages used vision)",
            len(chunks),
            vision_pages,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during ingestion")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        status="success",
        pages_processed=len(raw_pages),
        chunks_created=len(chunks),
        vision_pages=vision_pages,
        message=(
            f"Ingested {document_name}: "
            f"{len(raw_pages)} pages → {len(chunks)} chunks "
            f"({vision_pages} pages used vision fallback)."
        ),
    )


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query(request: QueryRequest) -> QueryResponse:
    """
    Answer an analyst-style question using evidence retrieved from the PDF.

    Returns a grounded answer with inline [Page X] citations, structured
    citation objects, and a limitations list when evidence is weak.
    """
    info = collection_info()
    if not info["exists"] or info["chunk_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been ingested yet. Call POST /ingest first.",
        )

    try:
        query_vector = embed_query(request.question)
        results = search(query_vector, top_k=request.top_k)
        answer, citations, limitations = generate_answer(request.question, results)
    except Exception as exc:
        logger.exception("Unexpected error during query")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    return QueryResponse(
        answer=answer,
        citations=citations,
        retrieval=RetrievalInfo(
            top_k=request.top_k,
            chunks_consulted=len(results),
        ),
        limitations=limitations,
    )
