"""Database models exposed for Alembic metadata discovery."""

from app.models.fitness import (
    AuditLog,
    BodyweightLog,
    Exercise,
    FoodLog,
    Member,
    MemberNutritionTarget,
    MemberProfile,
    MemberTrendFlag,
    Subscription,
    WorkoutLog,
    WorkoutPlan,
)

__all__ = [
    "AuditLog",
    "BodyweightLog",
    "Exercise",
    "FoodLog",
    "Member",
    "MemberNutritionTarget",
    "MemberProfile",
    "MemberTrendFlag",
    "Subscription",
    "WorkoutLog",
    "WorkoutPlan",
]

