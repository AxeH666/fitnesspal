"""Store recommended and active protein targets separately.

Revision ID: 20260830_0003
Revises: 20260830_0002
Create Date: 2026-08-30 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260830_0003"
down_revision: str | None = "20260830_0002"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Allow recommendation-only and active-only POC target state."""

    op.add_column(
        "member_nutrition_targets",
        sa.Column("recommended_protein_target_g", sa.Integer(), nullable=True),
    )
    for column_name in (
        "calorie_target_kcal",
        "protein_target_g",
        "fat_target_g",
        "carb_target_g",
    ):
        op.alter_column(
            "member_nutrition_targets",
            column_name,
            existing_type=sa.Integer(),
            nullable=True,
        )


def downgrade() -> None:
    """Restore Phase 0 constraints only when existing rows are compatible."""

    op.execute(
        sa.text(
            "DO $$ "
            "BEGIN "
            "IF EXISTS ("
            "SELECT 1 FROM member_nutrition_targets "
            "WHERE calorie_target_kcal IS NULL "
            "OR protein_target_g IS NULL "
            "OR fat_target_g IS NULL "
            "OR carb_target_g IS NULL"
            ") THEN "
            "RAISE EXCEPTION 'Cannot downgrade while nutrition targets contain NULL values'; "
            "END IF; "
            "END $$"
        )
    )
    for column_name in (
        "calorie_target_kcal",
        "protein_target_g",
        "fat_target_g",
        "carb_target_g",
    ):
        op.alter_column(
            "member_nutrition_targets",
            column_name,
            existing_type=sa.Integer(),
            nullable=False,
        )
    op.drop_column("member_nutrition_targets", "recommended_protein_target_g")
