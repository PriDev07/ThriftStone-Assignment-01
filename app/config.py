"""
Central configuration for the Investor Presentation RAG system.

All values are loaded from environment variables or a ``.env`` file.
Copy ``.env.example`` to ``.env`` and fill in ``GROQ_API_KEY`` to get started.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Groq API — required
    groq_api_key: str

    # Qdrant — local mode, no Docker required
    # Set qdrant_path to a directory path to persist data across restarts.
    qdrant_path: str = ":memory:"
    qdrant_collection: str = "investor_presentation"

    # Embedding model (sentence-transformers hub name)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Chunking
    chunk_size: int = 400    # approximate word count per chunk
    chunk_overlap: int = 80  # word overlap between adjacent chunks

    # Retrieval
    default_top_k: int = 5

    # LLM
    llm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
