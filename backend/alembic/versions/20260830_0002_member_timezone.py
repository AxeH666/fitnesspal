"""Store the member timezone used for local calendar dates.

Revision ID: 20260830_0002
Revises: 20260826_0001
Create Date: 2026-08-30 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260830_0002"
down_revision: str | None = "20260826_0001"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Add and backfill the IANA timezone required for chronological state."""

    op.add_column(
        "member_profiles",
        sa.Column("timezone", sa.String(length=64), nullable=True),
    )
    op.execute(
        sa.text(
            "UPDATE member_profiles "
            "SET timezone = 'Asia/Kolkata' "
            "WHERE timezone IS NULL"
        )
    )
    op.alter_column(
        "member_profiles",
        "timezone",
        existing_type=sa.String(length=64),
        nullable=False,
    )


def downgrade() -> None:
    """Remove the member timezone."""

    op.drop_column("member_profiles", "timezone")
