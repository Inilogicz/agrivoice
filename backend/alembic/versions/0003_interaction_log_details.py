"""Store prompts, responses, timings and errors on interaction_logs.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-01 00:00:00.000000
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_NEW_COLUMNS = [
    sa.Column("error_message", sa.Text(), nullable=True),
    sa.Column("prompt", sa.Text(), nullable=True),
    sa.Column("response", sa.Text(), nullable=True),
    sa.Column("language_source", sa.String(length=16), nullable=True),
    sa.Column("language_confidence", sa.Float(), nullable=True),
    sa.Column("llm_available", sa.Boolean(), nullable=True),
    sa.Column("detection_ms", sa.Integer(), nullable=True),
    sa.Column("asr_ms", sa.Integer(), nullable=True),
    sa.Column("llm_ms", sa.Integer(), nullable=True),
    sa.Column("request_id", sa.String(length=64), nullable=True),
]


def upgrade() -> None:
    for column in _NEW_COLUMNS:
        op.add_column("interaction_logs", column)
    op.alter_column("interaction_logs", "conversation_id", nullable=True)
    op.create_index("ix_interaction_logs_created_at", "interaction_logs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_interaction_logs_created_at", table_name="interaction_logs")
    op.alter_column("interaction_logs", "conversation_id", nullable=False)
    for column in reversed(_NEW_COLUMNS):
        op.drop_column("interaction_logs", column.name)
