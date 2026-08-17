"""FinSight retrieval package.

Phase 3 scope (PRD): dense embeddings + pgvector storage + similarity search.
"""

from src.retrieval.embeddings import (
    ChunkEmbedding,
    Embedder,
    build_embedder,
    embed_chunks,
)
from src.retrieval.search import load_chunk_files, retrieve_dense, seed_database
from src.retrieval.storage import (
    get_engine,
    init_schema,
    insert_chunks,
    similarity_search,
)

__all__ = [
    "ChunkEmbedding",
    "Embedder",
    "build_embedder",
    "embed_chunks",
    "get_engine",
    "init_schema",
    "insert_chunks",
    "load_chunk_files",
    "retrieve_dense",
    "seed_database",
    "similarity_search",
]