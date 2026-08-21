"""FinSight retrieval package.

Phase 3 scope (PRD): dense embeddings + pgvector storage + similarity search.
"""

from src.retrieval.bm25 import BM25Index, build_bm25_index
from src.retrieval.compress import compress_chunks, split_sentences
from src.retrieval.embeddings import (
    ChunkEmbedding,
    Embedder,
    build_embedder,
    embed_chunks,
)
from src.retrieval.hybrid import fuse_hybrid, retrieve_hybrid
from src.retrieval.rerank import (
    CrossEncoderReranker,
    build_reranker,
    rerank_candidates,
    retrieve_reranked,
)
from src.retrieval.search import load_chunk_files, retrieve_dense, seed_database
from src.retrieval.storage import (
    get_engine,
    init_schema,
    insert_chunks,
    similarity_search,
)

__all__ = [
    "BM25Index",
    "ChunkEmbedding",
    "CrossEncoderReranker",
    "Embedder",
    "build_bm25_index",
    "build_embedder",
    "build_reranker",
    "compress_chunks",
    "embed_chunks",
    "fuse_hybrid",
    "get_engine",
    "init_schema",
    "insert_chunks",
    "load_chunk_files",
    "rerank_candidates",
    "retrieve_dense",
    "retrieve_hybrid",
    "retrieve_reranked",
    "seed_database",
    "similarity_search",
    "split_sentences",
]