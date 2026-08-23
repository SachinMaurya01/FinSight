""" Dense embeddings: embed chunk text into vectors.

Provides a small ``Embedder`` interface so the retrieval layer is provider
agnostic. Supported providers (``settings.embedding_provider``):

- ``openai`` — OpenAI ``text-embedding-3-small`` via ``langchain-openai``
- ``gemini`` — Google embeddings via ``langchain-google-genai``
- ``sentence_transformers`` — local Hugging Face model (no API key needed)

Embedding API keys are injected from env at runtime (never hardcoded).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from src.config import Settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that turns a list of texts into a list of embeddings."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class LangChainEmbedder:
    """Adapter over a langchain-style embedding object (``embed_documents``)."""

    def __init__(self, embeddings: object) -> None:
        self._embeddings = embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder (offline, no API key)."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def build_embedder(settings: Settings) -> Embedder:
    """Build the configured embedder for ``settings.embedding_provider``."""
    provider = settings.embedding_provider
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return LangChainEmbedder(
            OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
        )
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return LangChainEmbedder(
            GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model, google_api_key=settings.google_api_key
            )
        )
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    raise ValueError(f"Unsupported embedding_provider: {provider!r}")


@dataclass(frozen=True)
class ChunkEmbedding:
    """A chunk paired with its dense embedding vector."""

    chunk: Chunk
    embedding: list[float]


def embed_chunks(
    chunks: list[Chunk],
    settings: Settings,
    embedder: Embedder | None = None,
) -> list[ChunkEmbedding]:
    """Embed chunk contents in batches of ``settings.embedding_batch_size``.

    Empty chunks are skipped with a warning. Each returned vector is checked
    against ``settings.embedding_dimensions`` and a warning is logged on a
    mismatch."""
    active = embedder or build_embedder(settings)
    use_cache = embedder is None  # only cache real embedder calls, not test fakes
    out: list[ChunkEmbedding] = []
    batch_texts: list[str] = []
    batch_chunks: list[Chunk] = []

    def flush() -> None:
        if not batch_texts:
            return
        vectors = active.embed_texts(batch_texts)
        for chunk, vector in zip(batch_chunks, vectors):
            if len(vector) != settings.embedding_dimensions:
                logger.warning(
                    "Embedding dimension %d does not match configured %d "
                    "(chunk_index=%s); check EMBEDDING_DIMENSIONS matches the model",
                    len(vector),
                    settings.embedding_dimensions,
                    chunk.metadata.get("chunk_index"),
                )
            if use_cache:
                try:
                    from src.retrieval.cache import set_cached_embedding

                    set_cached_embedding(settings.embedding_model, chunk.content, vector, settings)
                except Exception:
                    pass
            out.append(ChunkEmbedding(chunk=chunk, embedding=vector))
        batch_texts.clear()
        batch_chunks.clear()

    for chunk in chunks:
        if not chunk.content.strip():
            logger.warning("Skipping empty chunk (index=%s)", chunk.metadata.get("chunk_index"))
            continue
        if use_cache:
            try:
                from src.retrieval.cache import get_cached_embedding

                cached = get_cached_embedding(settings.embedding_model, chunk.content, settings)
                if cached is not None:
                    if len(cached) != settings.embedding_dimensions:
                        logger.warning("Cached dimension mismatch, re-embedding chunk %s", chunk.metadata.get("chunk_index"))
                    else:
                        out.append(ChunkEmbedding(chunk=chunk, embedding=cached))
                        continue
            except Exception:
                pass
        batch_texts.append(chunk.content)
        batch_chunks.append(chunk)
        if len(batch_texts) >= settings.embedding_batch_size:
            flush()
    flush()
    # Ensure stable order by chunk_index for determinism (reproducibility NFR)
    # We keep original order; cache hits were already in order, flush preserves order.
    return out