"""Date-scoped reads of persisted member fitness state."""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import json
import math
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.dates import local_date
from app.models import BodyweightLog, FoodLog, MemberProfile, WorkoutLog


@dataclass(frozen=True, slots=True)
class NutritionTotals:
    """Model-estimated nutrition for one meal or summed calendar day."""

    calories: int = 0
    protein_g: int = 0
    carbs_g: int = 0
    fat_g: int = 0


@dataclass(frozen=True, slots=True)
class DayState:
    """Fitness records belonging to one member and one local calendar date."""

    member_id: UUID
    local_date: date
    bodyweight_log: BodyweightLog | None
    food_logs: tuple[FoodLog, ...]
    food_totals: NutritionTotals
    workout_logs: tuple[WorkoutLog, ...]


def _member_local_date(session: Session, member_id: UUID, instant: datetime) -> date:
    """Resolve an instant using the member's persisted timezone."""

    timezone_name = session.scalar(
        select(MemberProfile.timezone).where(MemberProfile.member_id == member_id)
    )
    if timezone_name is None:
        raise LookupError(f"No timezone found for member {member_id}")

    return local_date(instant, timezone_name)


def get_bodyweight_for_day(
    session: Session,
    member_id: UUID,
    day: date,
) -> BodyweightLog | None:
    """Return the member's bodyweight for one local calendar date."""

    return session.scalar(
        select(BodyweightLog).where(
            BodyweightLog.member_id == member_id,
            BodyweightLog.log_date == day,
        )
    )


def get_latest_bodyweight(session: Session, member_id: UUID) -> BodyweightLog | None:
    """Return the member's bodyweight from their newest logged local date."""

    return session.scalar(
        select(BodyweightLog)
        .where(BodyweightLog.member_id == member_id)
        .order_by(
            BodyweightLog.log_date.desc(),
            BodyweightLog.recorded_at.desc(),
            BodyweightLog.id.desc(),
        )
        .limit(1)
    )


def log_bodyweight(
    session: Session,
    member_id: UUID,
    weight_kg: Decimal,
    event_time: datetime,
) -> BodyweightLog:
    """Create or replace the member's bodyweight for the event's local date."""

    if not weight_kg.is_finite() or weight_kg <= 0:
        raise ValueError("weight_kg must be a finite value greater than zero")

    day = _member_local_date(session, member_id, event_time)
    recorded_at = event_time.astimezone(timezone.utc)
    table = BodyweightLog.__table__
    insert_statement = insert(BodyweightLog).values(
        member_id=member_id,
        log_date=day,
        weight_kg=weight_kg,
        created_at=recorded_at,
    )
    upsert_statement = insert_statement.on_conflict_do_update(
        index_elements=[table.c.member_id, table.c.log_date],
        set_={
            table.c.weight_kg: insert_statement.excluded.weight_kg,
            table.c.created_at: insert_statement.excluded.created_at,
        },
    ).returning(BodyweightLog)

    return session.scalars(
        upsert_statement,
        execution_options={"populate_existing": True},
    ).one()


def _nutrition_totals_payload(totals: NutritionTotals) -> dict[str, int]:
    """Validate and serialize the four stored per-meal nutrition estimates."""

    payload = {
        "calories": totals.calories,
        "protein_g": totals.protein_g,
        "carbs_g": totals.carbs_g,
        "fat_g": totals.fat_g,
    }
    for field_name, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")

    return payload


def _valid_food_quantity(value: object) -> bool:
    """Return whether an item quantity is a positive finite JSON number."""

    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, float):
        return math.isfinite(value) and value > 0
    return False


def _validated_food_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy structured meal items after validating their minimum POC shape."""

    if not items:
        raise ValueError("items must contain at least one food item")

    copied_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        name: object = item.get("name")
        unit: object = item.get("unit")
        quantity: object = item.get("qty")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"items[{index}].name must be a non-empty string")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"items[{index}].unit must be a non-empty string")
        if not _valid_food_quantity(quantity):
            raise ValueError(f"items[{index}].qty must be a positive finite number")

        for nutrient in ("calories", "protein_g", "carbs_g", "fat_g"):
            if nutrient not in item:
                continue
            value: object = item[nutrient]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (isinstance(value, float) and not math.isfinite(value))
                or value < 0
            ):
                raise ValueError(
                    f"items[{index}].{nutrient} must be a non-negative finite number"
                )

        copied_item = dict(item)
        copied_item["name"] = name.strip()
        copied_item["unit"] = unit.strip()
        try:
            canonical_item = json.loads(json.dumps(copied_item, allow_nan=False))
        except (TypeError, ValueError):
            raise ValueError(
                f"items[{index}] must contain only finite JSON-compatible values"
            ) from None
        copied_items.append(canonical_item)

    return copied_items


def _validated_meal_type(meal_type: str | None) -> str | None:
    """Normalize the optional meal label within the existing schema limit."""

    if meal_type is None:
        return None
    normalized = meal_type.strip()
    if not normalized:
        raise ValueError("meal_type must be non-empty when provided")
    if len(normalized) > 20:
        raise ValueError("meal_type must be 20 characters or fewer")
    return normalized


def log_food(
    session: Session,
    member_id: UUID,
    items: list[dict[str, Any]],
    totals: NutritionTotals,
    event_time: datetime,
    *,
    meal_type: str | None = None,
    note: str | None = None,
) -> FoodLog:
    """Append one structured, model-estimated meal to its member-local day."""

    copied_items = _validated_food_items(items)
    totals_payload = _nutrition_totals_payload(totals)
    normalized_meal_type = _validated_meal_type(meal_type)
    day = _member_local_date(session, member_id, event_time)
    food_log = FoodLog(
        member_id=member_id,
        log_date=day,
        meal_type=normalized_meal_type,
        items=copied_items,
        totals=totals_payload,
        recorded_at=event_time.astimezone(timezone.utc),
        note=note,
    )
    session.add(food_log)
    session.flush()
    return food_log


def get_food_for_day(
    session: Session,
    member_id: UUID,
    day: date,
) -> tuple[FoodLog, ...]:
    """Return the member's meals for one local calendar date."""

    return tuple(
        session.scalars(
            select(FoodLog)
            .where(FoodLog.member_id == member_id, FoodLog.log_date == day)
            .order_by(FoodLog.recorded_at, FoodLog.id)
        ).all()
    )


def _stored_nutrition_value(totals: dict[str, Any], field_name: str) -> int:
    """Read one validated nutrition value from a persisted meal."""

    value: object = totals.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Stored food total {field_name} must be a non-negative integer"
        )
    return value


def _sum_food_totals(food_logs: tuple[FoodLog, ...]) -> NutritionTotals:
    """Calculate authoritative daily totals from persisted per-meal estimates."""

    calories = 0
    protein_g = 0
    carbs_g = 0
    fat_g = 0
    for food_log in food_logs:
        calories += _stored_nutrition_value(food_log.totals, "calories")
        protein_g += _stored_nutrition_value(food_log.totals, "protein_g")
        carbs_g += _stored_nutrition_value(food_log.totals, "carbs_g")
        fat_g += _stored_nutrition_value(food_log.totals, "fat_g")

    return NutritionTotals(
        calories=calories,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
    )


def get_food_totals_for_day(
    session: Session,
    member_id: UUID,
    day: date,
) -> NutritionTotals:
    """Return deterministic nutrition totals for one member-local date."""

    return _sum_food_totals(get_food_for_day(session, member_id, day))


def get_today_food_totals(
    session: Session,
    member_id: UUID,
    *,
    now: datetime | None = None,
) -> NutritionTotals:
    """Return today's nutrition totals in the member's persisted timezone."""

    instant = now if now is not None else datetime.now(timezone.utc)
    day = _member_local_date(session, member_id, instant)
    return get_food_totals_for_day(session, member_id, day)


def get_day_state(session: Session, member_id: UUID, day: date) -> DayState:
    """Return only records matching the requested member and local date."""

    bodyweight_log = get_bodyweight_for_day(session, member_id, day)
    food_logs = get_food_for_day(session, member_id, day)
    food_totals = _sum_food_totals(food_logs)
    workout_logs = tuple(
        session.scalars(
            select(WorkoutLog)
            .where(WorkoutLog.member_id == member_id, WorkoutLog.log_date == day)
            .order_by(WorkoutLog.created_at, WorkoutLog.id)
        ).all()
    )

    return DayState(
        member_id=member_id,
        local_date=day,
        bodyweight_log=bodyweight_log,
        food_logs=food_logs,
        food_totals=food_totals,
        workout_logs=workout_logs,
    )


def get_today_state(
    session: Session,
    member_id: UUID,
    *,
    now: datetime | None = None,
) -> DayState:
    """Return the member's state for today in their persisted timezone."""

    instant = now if now is not None else datetime.now(timezone.utc)
    return get_day_state(session, member_id, _member_local_date(session, member_id, instant))
