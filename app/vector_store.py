"""
Qdrant vector store interface — local / in-memory mode (no Docker required).

``QdrantClient(path=":memory:")`` runs entirely in-process with zero
infrastructure. Set ``QDRANT_PATH=./qdrant_data`` in ``.env`` to persist
the index to disk across server restarts.

Switching to Qdrant Cloud later requires only changing the client
constructor from ``path=`` to ``url=`` / ``host=``.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    """
    Return a cached QdrantClient instance.

    The client is created once per process and reused for all operations.
    ``path=":memory:"`` keeps the index in RAM; ``path="./qdrant_data"``
    persists it to disk.
    """
    return QdrantClient(path=settings.qdrant_path)


def ensure_collection(recreate: bool = False) -> None:
    """
    Create the Qdrant collection if it does not already exist.

    Parameters
    ----------
    recreate : bool
        When True, delete the existing collection before creating a fresh one.
        Use this to guarantee no stale chunks from a previous ingestion run.
    """
    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]

    if recreate and settings.qdrant_collection in existing:
        client.delete_collection(settings.qdrant_collection)
        existing = []

    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qmodels.VectorParams(
                size=settings.embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )


def _chunk_id_to_point_id(chunk_id: str) -> int:
    """
    Convert a chunk_id string to a stable uint64 Qdrant point ID.

    Uses the first 8 bytes of SHA-256(chunk_id) for a deterministic,
    collision-resistant mapping. Python's built-in ``hash()`` is
    process-salted and must not be used here.

    Parameters
    ----------
    chunk_id : str
        Hex chunk identifier produced by the chunker.

    Returns
    -------
    int
        Unsigned 64-bit integer suitable as a Qdrant point ID.
    """
    digest = hashlib.sha256(chunk_id.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def upsert_chunks(chunks, vectors: List[List[float]]) -> None:
    """
    Upsert chunk payloads and their embedding vectors into Qdrant.

    Batches upserts in groups of 256 to avoid oversized request payloads.

    Parameters
    ----------
    chunks : list of Chunk
        Chunk objects produced by the ingestion pipeline.
    vectors : list of list of float
        Embedding vectors corresponding to each chunk, in the same order.
    """
    client = _get_client()

    points = [
        qmodels.PointStruct(
            id=_chunk_id_to_point_id(chunk.chunk_id),
            vector=vector,
            payload={
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                **chunk.metadata,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    batch_size = 256
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[i: i + batch_size],
        )


def search(
    query_vector: List[float],
    top_k: int = 5,
) -> List[Tuple[float, dict]]:
    """
    Return the top-k most similar chunks for a query vector.

    Parameters
    ----------
    query_vector : list of float
        Embedding of the user's question (with BGE query prefix applied).
    top_k : int
        Number of results to return.

    Returns
    -------
    list of (score, payload) tuples
        Ordered by descending cosine similarity score.
    """
    client = _get_client()
    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [(hit.score, hit.payload) for hit in results]


def collection_info() -> dict:
    """
    Return basic statistics about the Qdrant collection.

    Returns
    -------
    dict
        ``{"exists": bool, "chunk_count": int}``
    """
    client = _get_client()
    try:
        info = client.get_collection(settings.qdrant_collection)
        return {"exists": True, "chunk_count": info.points_count}
    except Exception:
        return {"exists": False, "chunk_count": 0}


def qdrant_healthy() -> bool:
    """
    Return True if the Qdrant client is reachable and operational.

    Returns
    -------
    bool
    """
    try:
        _get_client().get_collections()
        return True
    except Exception:
        return False
