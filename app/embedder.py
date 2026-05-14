"""
Embedding model wrapper — BAAI/bge-small-en-v1.5.

Why BGE over all-MiniLM-L6-v2:
  - 512-token context window (vs 256) — fewer chunks are silently truncated
    during embedding, which matters for dense financial slides.
  - Trained specifically for retrieval tasks; outperforms MiniLM on BEIR
    passage-retrieval benchmarks.
  - Same 384-dimensional output — no change to the Qdrant collection config.
  - Runs fully locally; no API key or rate limits.

BGE query prefix:
  BGE models are asymmetric: queries and documents must be encoded differently.
  Queries require the prefix below; document chunks are encoded without it.
  Omitting the prefix on queries measurably degrades retrieval quality.
  See: https://huggingface.co/BAAI/bge-small-en-v1.5
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import settings

# BGE models require this prefix for query embeddings (not document embeddings).
# See: https://huggingface.co/BAAI/bge-small-en-v1.5
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it for the process lifetime."""
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of document chunks without any prefix.

    Used during ingestion. Documents are encoded as-is per the BGE spec.

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
    Embed a retrieval query with the required BGE prefix.

    Used at query time. The prefix is prepended before encoding to align
    the query representation with the document embedding space.

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
    # BGE models require this prefix for query embeddings (not document embeddings)
    # See: https://huggingface.co/BAAI/bge-small-en-v1.5
    prefixed = _BGE_QUERY_PREFIX + query
    vector = model.encode(prefixed, show_progress_bar=False, convert_to_numpy=True)
    return vector.tolist()
