"""Phase 7 API keys and HITL audit log.

Revision ID: 0002_security_phase7
Revises: 0001_chunks_phase5
Create Date: 2026-11-13
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_security_phase7"
down_revision = "0001_chunks_phase5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("key_hash", sa.Text(), nullable=False, unique=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("request_limit_per_minute", sa.Integer()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "request_limit_per_minute IS NULL OR request_limit_per_minute > 0",
            name="api_keys_limit_positive",
        ),
    )
    op.create_index("api_keys_active_idx", "api_keys", ["is_active"])

    op.create_table(
        "hitl_audit",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("reviewer", sa.Text(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("note", sa.Text()),
        sa.Column("draft_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("decision IN ('approved', 'rejected')", name="hitl_audit_decision"),
    )
    op.create_index("hitl_audit_thread_idx", "hitl_audit", ["thread_id", "created_at"])
    op.create_index("hitl_audit_created_idx", "hitl_audit", ["created_at"])


def downgrade() -> None:
    op.drop_index("hitl_audit_created_idx", table_name="hitl_audit")
    op.drop_index("hitl_audit_thread_idx", table_name="hitl_audit")
    op.drop_table("hitl_audit")
    op.drop_index("api_keys_active_idx", table_name="api_keys")
    op.drop_table("api_keys")
