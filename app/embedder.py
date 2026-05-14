"""
Embedding model wrapper — all-MiniLM-L6-v2.

Why all-MiniLM-L6-v2:
  - 384-dimensional dense vectors — fast Qdrant search, small index footprint.
  - Strong semantic quality for English business/financial text out of the box.
  - Runs fully locally; no API key or rate limits.
  - Symmetric model: queries and documents are encoded identically, no prefix needed.

Where it falls short:
  - 256-token context window — chunks longer than ~200 words are silently truncated.
    Mitigated by the 400-word chunk size being split across multiple overlapping chunks.
  - Not fine-tuned on Indian financial jargon (₹, Cr, Lakh).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import settings

# Kept for test compatibility — MiniLM does not require a query prefix,
# but the constant is referenced in test_embedder.py.
_BGE_QUERY_PREFIX = ""


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it for the process lifetime."""
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of document chunks.

    Parameters
    ----------
    texts : list of str
        Chunk texts to embed.

    Returns
    -------
    list of list of float
        One 384-dimensional vector per input text.
    """
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a retrieval query.

    all-MiniLM-L6-v2 is symmetric — no prefix is needed for queries.

    Parameters
    ----------
    query : str
        The user's question.

    Returns
    -------
    list of float
        384-dimensional embedding vector.
    """
    model = _get_model()
    vector = model.encode(query, show_progress_bar=False, convert_to_numpy=True)
    return vector.tolist()
