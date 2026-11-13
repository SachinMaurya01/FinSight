"""Phase 5 chunks metadata schema.

Revision ID: 0001_chunks_phase5
Revises:
Create Date: 2026-11-13
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from src.config import settings

revision = "0001_chunks_phase5"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(settings.embedding_dimensions)),
        sa.Column("ticker", sa.Text()),
        sa.Column("cik", sa.BigInteger()),
        sa.Column("filing_type", sa.Text()),
        sa.Column("form_type", sa.Text()),
        sa.Column("fiscal_period", sa.Text()),
        sa.Column("fiscal_year", sa.Integer()),
        sa.Column("section", sa.Text()),
        sa.Column("chunk_type", sa.Text(), nullable=False, server_default="text"),
        sa.Column("accession", sa.Text()),
        sa.Column("filed_at", sa.Text()),
        sa.Column("source_doc", sa.Text()),
        sa.Column("chunk_index", sa.Integer()),
        sa.Column("char_offset", sa.Integer()),
        sa.Column("embedding_model", sa.Text()),
        sa.Column("embedding_version", sa.Text()),
        sa.Column("metadata", postgresql.JSONB()),
    )
    op.create_index(
        "chunks_embedding_idx",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("chunks_meta_idx", "chunks", ["ticker", "fiscal_year", "form_type"])
    op.create_index("chunks_accession_chunk_idx", "chunks", ["accession", "chunk_index"], unique=True)


def downgrade() -> None:
    op.drop_index("chunks_accession_chunk_idx", table_name="chunks")
    op.drop_index("chunks_meta_idx", table_name="chunks")
    op.drop_index("chunks_embedding_idx", table_name="chunks")
    op.drop_table("chunks")
