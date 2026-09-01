"""Date-scoped reads of persisted member fitness state."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
import math
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.dates import local_date
from app.models import BodyweightLog, Exercise, FoodLog, MemberProfile, WorkoutLog


@dataclass(frozen=True, slots=True)
class NutritionTotals:
    """Model-estimated nutrition for one meal or summed calendar day."""

    calories: int = 0
    protein_g: int = 0
    carbs_g: int = 0
    fat_g: int = 0


@dataclass(frozen=True, slots=True)
class WorkoutSetInput:
    """One completed set supplied by a deterministic workout operation."""

    reps: int
    weight_kg: int | float


@dataclass(frozen=True, slots=True)
class WorkoutExerciseInput:
    """One canonical exercise and its completed sets."""

    exercise_id: UUID
    sets: tuple[WorkoutSetInput, ...]


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


def _workout_exercises_payload(
    session: Session,
    exercises: tuple[WorkoutExerciseInput, ...],
) -> list[dict[str, Any]]:
    """Validate workout performance and attach canonical exercise names."""

    if not exercises:
        raise ValueError("exercises must contain at least one exercise")

    exercise_ids: list[UUID] = []
    serialized_sets: list[list[dict[str, int | float]]] = []
    for exercise_index, exercise in enumerate(exercises):
        if not isinstance(exercise, WorkoutExerciseInput):
            raise ValueError(
                f"exercises[{exercise_index}] must be a WorkoutExerciseInput"
            )
        if not isinstance(exercise.exercise_id, UUID):
            raise ValueError(f"exercises[{exercise_index}].exercise_id must be a UUID")
        if not exercise.sets:
            raise ValueError(
                f"exercises[{exercise_index}].sets must contain at least one set"
            )

        set_payloads: list[dict[str, int | float]] = []
        for set_index, completed_set in enumerate(exercise.sets):
            if not isinstance(completed_set, WorkoutSetInput):
                raise ValueError(
                    f"exercises[{exercise_index}].sets[{set_index}] "
                    "must be a WorkoutSetInput"
                )
            if (
                isinstance(completed_set.reps, bool)
                or not isinstance(completed_set.reps, int)
                or completed_set.reps <= 0
            ):
                raise ValueError(
                    f"exercises[{exercise_index}].sets[{set_index}].reps "
                    "must be a positive integer"
                )

            weight_kg = completed_set.weight_kg
            if (
                isinstance(weight_kg, bool)
                or not isinstance(weight_kg, (int, float))
                or (isinstance(weight_kg, float) and not math.isfinite(weight_kg))
                or weight_kg < 0
            ):
                raise ValueError(
                    f"exercises[{exercise_index}].sets[{set_index}].weight_kg "
                    "must be a non-negative finite number"
                )

            set_payloads.append({"reps": completed_set.reps, "weight_kg": weight_kg})

        exercise_ids.append(exercise.exercise_id)
        serialized_sets.append(set_payloads)

    stored_exercises = {
        exercise_id: name
        for exercise_id, name in session.execute(
            select(Exercise.id, Exercise.name).where(Exercise.id.in_(exercise_ids))
        ).all()
    }
    missing_exercise_ids = tuple(
        exercise_id
        for exercise_id in dict.fromkeys(exercise_ids)
        if exercise_id not in stored_exercises
    )
    if missing_exercise_ids:
        missing = ", ".join(str(exercise_id) for exercise_id in missing_exercise_ids)
        raise LookupError(f"Unknown exercise IDs: {missing}")

    return [
        {
            "exercise_id": str(exercise.exercise_id),
            "name": stored_exercises[exercise.exercise_id],
            "sets": set_payloads,
        }
        for exercise, set_payloads in zip(exercises, serialized_sets, strict=True)
    ]


def log_workout(
    session: Session,
    member_id: UUID,
    exercises: tuple[WorkoutExerciseInput, ...],
    event_time: datetime,
    *,
    note: str | None = None,
) -> WorkoutLog:
    """Append one completed workout to its member-local calendar date."""

    day = _member_local_date(session, member_id, event_time)
    exercises_payload = _workout_exercises_payload(session, exercises)
    workout_log = WorkoutLog(
        member_id=member_id,
        log_date=day,
        exercises=exercises_payload,
        recorded_at=event_time.astimezone(timezone.utc),
        note=note,
    )
    session.add(workout_log)
    session.flush()
    return workout_log


def get_workouts_for_day(
    session: Session,
    member_id: UUID,
    day: date,
) -> tuple[WorkoutLog, ...]:
    """Return one member's workouts for one local calendar date."""

    return tuple(
        session.scalars(
            select(WorkoutLog)
            .where(WorkoutLog.member_id == member_id, WorkoutLog.log_date == day)
            .order_by(WorkoutLog.recorded_at, WorkoutLog.id)
        ).all()
    )


def _validated_workout_history_limit(limit: int) -> int:
    """Require an explicit positive bound for workout-history reads."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    return limit


def get_recent_workouts(
    session: Session,
    member_id: UUID,
    *,
    limit: int = 10,
) -> tuple[WorkoutLog, ...]:
    """Return a member's most recent workouts in reverse chronological order."""

    validated_limit = _validated_workout_history_limit(limit)
    return tuple(
        session.scalars(
            select(WorkoutLog)
            .where(WorkoutLog.member_id == member_id)
            .order_by(
                WorkoutLog.log_date.desc(),
                WorkoutLog.recorded_at.desc(),
                WorkoutLog.id.desc(),
            )
            .limit(validated_limit)
        ).all()
    )


def _workout_contains_exercise(workout: WorkoutLog, exercise_id: UUID) -> bool:
    """Return whether one stored workout contains the canonical exercise ID."""

    expected_id = str(exercise_id)
    return any(
        isinstance(exercise, dict) and exercise.get("exercise_id") == expected_id
        for exercise in workout.exercises
    )


def get_exercise_history(
    session: Session,
    member_id: UUID,
    exercise_id: UUID,
    *,
    limit: int = 10,
) -> tuple[WorkoutLog, ...]:
    """Return recent member workouts containing one canonical exercise."""

    if not isinstance(exercise_id, UUID):
        raise ValueError("exercise_id must be a UUID")
    validated_limit = _validated_workout_history_limit(limit)
    ordered_workouts = session.scalars(
        select(WorkoutLog)
        .where(WorkoutLog.member_id == member_id)
        .order_by(
            WorkoutLog.log_date.desc(),
            WorkoutLog.recorded_at.desc(),
            WorkoutLog.id.desc(),
        )
    )

    matching_workouts: list[WorkoutLog] = []
    for workout in ordered_workouts:
        if _workout_contains_exercise(workout, exercise_id):
            matching_workouts.append(workout)
            if len(matching_workouts) == validated_limit:
                break
    return tuple(matching_workouts)


def get_day_state(session: Session, member_id: UUID, day: date) -> DayState:
    """Return only records matching the requested member and local date."""

    bodyweight_log = get_bodyweight_for_day(session, member_id, day)
    food_logs = get_food_for_day(session, member_id, day)
    food_totals = _sum_food_totals(food_logs)
    workout_logs = get_workouts_for_day(session, member_id, day)

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


def get_yesterday_state(
    session: Session,
    member_id: UUID,
    *,
    now: datetime | None = None,
) -> DayState:
    """Return the prior member-local calendar day's persisted state."""

    instant = now if now is not None else datetime.now(timezone.utc)
    yesterday = _member_local_date(session, member_id, instant) - timedelta(days=1)
    return get_day_state(session, member_id, yesterday)
