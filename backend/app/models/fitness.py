"""Phase 0 SQLAlchemy models for the Barbarik Fitness Pal schema."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, CheckConstraint, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class UUIDPrimaryKeyMixin:
    """Provide a UUID primary key with both application and database defaults."""

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )


class CreatedAtMixin:
    """Provide a database-generated creation timestamp."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class CreatedUpdatedAtMixin(CreatedAtMixin):
    """Provide database-generated creation and update timestamps."""

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Member(UUIDPrimaryKeyMixin, CreatedUpdatedAtMixin, Base):
    """Gym member authentication and core profile data."""

    __tablename__ = "members"

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(100))
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    sex: Mapped[str | None] = mapped_column(String(10))
    height_cm: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))


class Subscription(UUIDPrimaryKeyMixin, CreatedUpdatedAtMixin, Base):
    """Membership subscription and payment reference."""

    __tablename__ = "subscriptions"

    member_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("members.id", ondelete="CASCADE")
    )
    tier: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'trial'"))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    payment_provider: Mapped[str | None] = mapped_column(String(50))
    payment_ref: Mapped[str | None] = mapped_column(String(255))
    grace_period_days: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("7"))


class MemberProfile(CreatedUpdatedAtMixin, Base):
    """Onboarding answers and contact preferences for one member."""

    __tablename__ = "member_profiles"
    __table_args__ = (
        CheckConstraint(
            "preferred_days_per_week BETWEEN 3 AND 6",
            name="ck_member_profiles_preferred_days_per_week",
        ),
    )

    member_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("members.id", ondelete="CASCADE"),
        primary_key=True,
    )
    goal: Mapped[str] = mapped_column(String(30), nullable=False)
    activity_level: Mapped[str] = mapped_column(String(20), nullable=False)
    training_experience: Mapped[str] = mapped_column(String(20), nullable=False)
    injuries_limitations: Mapped[str | None] = mapped_column(Text)
    preferred_days_per_week: Mapped[int | None] = mapped_column(Integer)
    preferred_split: Mapped[str | None] = mapped_column(String(30))
    dietary_preferences: Mapped[str | None] = mapped_column(Text)
    unit_system: Mapped[str] = mapped_column(String(10), nullable=False, server_default=text("'metric'"))
    whatsapp_linked: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    whatsapp_phone: Mapped[str | None] = mapped_column(String(20))
    whatsapp_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    onboarding_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemberNutritionTarget(CreatedUpdatedAtMixin, Base):
    """Calculated daily energy and macronutrient targets."""

    __tablename__ = "member_nutrition_targets"

    member_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("members.id", ondelete="CASCADE"),
        primary_key=True,
    )
    bmr_kcal: Mapped[Decimal | None] = mapped_column(Numeric(7, 2))
    tdee_kcal: Mapped[Decimal | None] = mapped_column(Numeric(7, 2))
    calorie_target_kcal: Mapped[int] = mapped_column(Integer, nullable=False)
    protein_target_g: Mapped[int] = mapped_column(Integer, nullable=False)
    fat_target_g: Mapped[int] = mapped_column(Integer, nullable=False)
    carb_target_g: Mapped[int] = mapped_column(Integer, nullable=False)
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class BodyweightLog(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """One bodyweight observation per member per calendar day."""

    __tablename__ = "bodyweight_logs"
    __table_args__ = (UniqueConstraint("member_id", "log_date", name="uq_bodyweight_logs_member_date"),)

    member_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("members.id", ondelete="CASCADE")
    )
    log_date: Mapped[date] = mapped_column(Date, nullable=False)
    weight_kg: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'whatsapp'"))
    note: Mapped[str | None] = mapped_column(Text)


class FoodLog(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """A member meal, including extracted food items and calculated totals."""

    __tablename__ = "food_logs"
    __table_args__ = (Index("idx_food_logs_member_date", "member_id", "log_date"),)

    member_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("members.id", ondelete="CASCADE")
    )
    log_date: Mapped[date] = mapped_column(Date, nullable=False)
    meal_type: Mapped[str | None] = mapped_column(String(20))
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    totals: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'whatsapp'"))
    note: Mapped[str | None] = mapped_column(Text)


class Exercise(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Canonical exercise library used by workouts and fuzzy matching."""

    __tablename__ = "exercises"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    aliases: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    primary_muscle: Mapped[str | None] = mapped_column(String(50))
    secondary_muscles: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    equipment: Mapped[str | None] = mapped_column(String(50))
    movement_pattern: Mapped[str | None] = mapped_column(String(30))
    is_compound: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))


class WorkoutPlan(UUIDPrimaryKeyMixin, CreatedUpdatedAtMixin, Base):
    """A named training plan and its JSON schedule."""

    __tablename__ = "workout_plans"

    member_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("members.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    split_type: Mapped[str] = mapped_column(String(30), nullable=False)
    days_per_week: Mapped[int] = mapped_column(Integer, nullable=False)
    schedule: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))


class WorkoutLog(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """A completed workout with exercises and per-set performance."""

    __tablename__ = "workout_logs"
    __table_args__ = (Index("idx_workout_logs_member_date", "member_id", "log_date"),)

    member_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("members.id", ondelete="CASCADE")
    )
    log_date: Mapped[date] = mapped_column(Date, nullable=False)
    plan_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("workout_plans.id"))
    exercises: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'whatsapp'"))
    note: Mapped[str | None] = mapped_column(Text)


class MemberTrendFlag(UUIDPrimaryKeyMixin, Base):
    """A deterministic, computed member trend signal."""

    __tablename__ = "member_trend_flags"
    __table_args__ = (Index("idx_trend_flags_member_active", "member_id", "is_active"),)

    member_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("members.id", ondelete="CASCADE")
    )
    flag_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))


class AuditLog(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Append-only mutation audit record."""

    __tablename__ = "audit_logs"
    __table_args__ = (Index("idx_audit_member_created", "member_id", "created_at"),)

    member_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("members.id", ondelete="SET NULL")
    )
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(30))
    entity_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    source: Mapped[str | None] = mapped_column(String(20))
    actor: Mapped[str | None] = mapped_column(String(50))

