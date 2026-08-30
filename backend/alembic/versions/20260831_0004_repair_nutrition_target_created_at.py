"""Repair missing nutrition-target creation timestamps.

Revision ID: 20260831_0004
Revises: 20260830_0003
Create Date: 2026-08-31 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260831_0004"
down_revision: str | None = "20260830_0003"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Repair databases whose recorded migration state omitted this column."""

    op.execute(
        sa.text(
            "ALTER TABLE member_nutrition_targets "
            "ADD COLUMN IF NOT EXISTS created_at "
            "TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()"
        )
    )


def downgrade() -> None:
    """Retain the column already owned by canonical revision 20260826_0001."""
