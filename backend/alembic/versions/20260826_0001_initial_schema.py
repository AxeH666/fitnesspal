"""Create the Phase 0 Barbarik Fitness Pal schema.

Revision ID: 20260826_0001
Revises:
Create Date: 2026-08-26 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260826_0001"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def uuid_pk() -> sa.Column[object]:
    """Build a PostgreSQL UUID primary key with a server-side default."""

    return sa.Column(
        "id",
        postgresql.UUID(as_uuid=True),
        primary_key=True,
        nullable=False,
        server_default=sa.text("gen_random_uuid()"),
    )


def timestamp_columns(*, updated: bool = False) -> list[sa.Column[object]]:
    """Return standard timestamps matching the MVP schema."""

    columns: list[sa.Column[object]] = [
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()"))
    ]
    if updated:
        columns.append(
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()"))
        )
    return columns


def upgrade() -> None:
    """Create extensions, Phase 0 tables, constraints, and indexes."""

    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "members",
        uuid_pk(),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("phone", sa.String(20), nullable=True, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(100), nullable=True),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("sex", sa.String(10), nullable=True),
        sa.Column("height_cm", sa.Numeric(5, 2), nullable=True),
        *timestamp_columns(updated=True),
    )
    op.create_table(
        "subscriptions",
        uuid_pk(),
        sa.Column("member_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("members.id", ondelete="CASCADE"), nullable=True),
        sa.Column("tier", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'trial'")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payment_provider", sa.String(50), nullable=True),
        sa.Column("payment_ref", sa.String(255), nullable=True),
        sa.Column("grace_period_days", sa.Integer(), nullable=False, server_default=sa.text("7")),
        *timestamp_columns(updated=True),
    )
    op.create_table(
        "member_profiles",
        sa.Column("member_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("members.id", ondelete="CASCADE"), primary_key=True, nullable=False),
        sa.Column("goal", sa.String(30), nullable=False),
        sa.Column("activity_level", sa.String(20), nullable=False),
        sa.Column("training_experience", sa.String(20), nullable=False),
        sa.Column("injuries_limitations", sa.Text(), nullable=True),
        sa.Column("preferred_days_per_week", sa.Integer(), nullable=True),
        sa.Column("preferred_split", sa.String(30), nullable=True),
        sa.Column("dietary_preferences", sa.Text(), nullable=True),
        sa.Column("unit_system", sa.String(10), nullable=False, server_default=sa.text("'metric'")),
        sa.Column("whatsapp_linked", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("whatsapp_phone", sa.String(20), nullable=True),
        sa.Column("whatsapp_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("onboarding_completed_at", sa.DateTime(timezone=True), nullable=True),
        *timestamp_columns(updated=True),
        sa.CheckConstraint("preferred_days_per_week BETWEEN 3 AND 6", name="ck_member_profiles_preferred_days_per_week"),
    )
    op.create_table(
        "member_nutrition_targets",
        sa.Column("member_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("members.id", ondelete="CASCADE"), primary_key=True, nullable=False),
        sa.Column("bmr_kcal", sa.Numeric(7, 2), nullable=True),
        sa.Column("tdee_kcal", sa.Numeric(7, 2), nullable=True),
        sa.Column("calorie_target_kcal", sa.Integer(), nullable=False),
        sa.Column("protein_target_g", sa.Integer(), nullable=False),
        sa.Column("fat_target_g", sa.Integer(), nullable=False),
        sa.Column("carb_target_g", sa.Integer(), nullable=False),
        sa.Column("calculated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        *timestamp_columns(updated=True),
    )
    op.create_table(
        "bodyweight_logs",
        uuid_pk(),
        sa.Column("member_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("members.id", ondelete="CASCADE"), nullable=True),
        sa.Column("log_date", sa.Date(), nullable=False),
        sa.Column("weight_kg", sa.Numeric(5, 2), nullable=False),
        sa.Column("source", sa.String(20), nullable=False, server_default=sa.text("'whatsapp'")),
        sa.Column("note", sa.Text(), nullable=True),
        *timestamp_columns(),
        sa.UniqueConstraint("member_id", "log_date", name="uq_bodyweight_logs_member_date"),
    )
    op.create_table(
        "food_logs",
        uuid_pk(),
        sa.Column("member_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("members.id", ondelete="CASCADE"), nullable=True),
        sa.Column("log_date", sa.Date(), nullable=False),
        sa.Column("meal_type", sa.String(20), nullable=True),
        sa.Column("items", postgresql.JSONB(), nullable=False),
        sa.Column("totals", postgresql.JSONB(), nullable=False),
        sa.Column("source", sa.String(20), nullable=False, server_default=sa.text("'whatsapp'")),
        sa.Column("note", sa.Text(), nullable=True),
        *timestamp_columns(),
    )
    op.create_index("idx_food_logs_member_date", "food_logs", ["member_id", "log_date"], unique=False)
    op.create_table(
        "exercises",
        uuid_pk(),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("aliases", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("primary_muscle", sa.String(50), nullable=True),
        sa.Column("secondary_muscles", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("equipment", sa.String(50), nullable=True),
        sa.Column("movement_pattern", sa.String(30), nullable=True),
        sa.Column("is_compound", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        *timestamp_columns(),
    )
    op.create_table(
        "workout_plans",
        uuid_pk(),
        sa.Column("member_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("members.id", ondelete="CASCADE"), nullable=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("split_type", sa.String(30), nullable=False),
        sa.Column("days_per_week", sa.Integer(), nullable=False),
        sa.Column("schedule", postgresql.JSONB(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        *timestamp_columns(updated=True),
    )
    op.create_table(
        "workout_logs",
        uuid_pk(),
        sa.Column("member_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("members.id", ondelete="CASCADE"), nullable=True),
        sa.Column("log_date", sa.Date(), nullable=False),
        sa.Column("plan_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workout_plans.id"), nullable=True),
        sa.Column("exercises", postgresql.JSONB(), nullable=False),
        sa.Column("source", sa.String(20), nullable=False, server_default=sa.text("'whatsapp'")),
        sa.Column("note", sa.Text(), nullable=True),
        *timestamp_columns(),
    )
    op.create_index("idx_workout_logs_member_date", "workout_logs", ["member_id", "log_date"], unique=False)
    op.create_table(
        "member_trend_flags",
        uuid_pk(),
        sa.Column("member_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("members.id", ondelete="CASCADE"), nullable=True),
        sa.Column("flag_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.create_index("idx_trend_flags_member_active", "member_trend_flags", ["member_id", "is_active"], unique=False)
    op.create_table(
        "audit_logs",
        uuid_pk(),
        sa.Column("member_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("members.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("entity_type", sa.String(30), nullable=True),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("before", postgresql.JSONB(), nullable=True),
        sa.Column("after", postgresql.JSONB(), nullable=True),
        sa.Column("source", sa.String(20), nullable=True),
        sa.Column("actor", sa.String(50), nullable=True),
        *timestamp_columns(),
    )
    op.create_index("idx_audit_member_created", "audit_logs", ["member_id", "created_at"], unique=False)


def downgrade() -> None:
    """Drop Phase 0 tables in dependency order; extensions remain installed."""

    op.drop_index("idx_audit_member_created", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index("idx_trend_flags_member_active", table_name="member_trend_flags")
    op.drop_table("member_trend_flags")
    op.drop_index("idx_workout_logs_member_date", table_name="workout_logs")
    op.drop_table("workout_logs")
    op.drop_table("workout_plans")
    op.drop_table("exercises")
    op.drop_index("idx_food_logs_member_date", table_name="food_logs")
    op.drop_table("food_logs")
    op.drop_table("bodyweight_logs")
    op.drop_table("member_nutrition_targets")
    op.drop_table("member_profiles")
    op.drop_table("subscriptions")
    op.drop_table("members")

