"""Explicit member-bound operations for the future POC agent."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Final, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, validate_call
from sqlalchemy.orm import Session

from app.models import BodyweightLog, FoodLog, WorkoutLog
from app.services import member_state, protein_targets


_HISTORY_LIMIT: Final = 10
_ResultT = TypeVar("_ResultT")


class _ToolValue(BaseModel):
    """Schema-ready value passed through the agent boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class NutritionTotalsValue(_ToolValue):
    """Four model-estimated meal totals or their deterministic daily sum."""

    calories: StrictInt
    protein_g: StrictInt
    carbs_g: StrictInt
    fat_g: StrictInt


class WorkoutSetValue(_ToolValue):
    """Agent-facing structured input for one completed workout set."""

    reps: StrictInt
    weight_kg: StrictInt | StrictFloat


class WorkoutExerciseValue(_ToolValue):
    """Agent-facing canonical exercise ID and completed sets."""

    exercise_id: UUID
    sets: tuple[WorkoutSetValue, ...]


class BodyweightResult(_ToolValue):
    """JSON-serializable bodyweight record returned to the agent."""

    local_date: date
    weight_kg: Decimal
    recorded_at: datetime
    note: str | None


class FoodLogResult(_ToolValue):
    """JSON-serializable stored meal returned to the agent."""

    local_date: date
    meal_type: str | None
    items: list[dict[str, Any]]
    totals: NutritionTotalsValue
    recorded_at: datetime
    note: str | None


class WorkoutLogResult(_ToolValue):
    """JSON-serializable stored workout returned to the agent."""

    local_date: date
    exercises: list[dict[str, Any]]
    recorded_at: datetime
    note: str | None


class DayStateResult(_ToolValue):
    """Stored fitness records for one member-local calendar day."""

    local_date: date
    bodyweight: BodyweightResult | None
    food_logs: tuple[FoodLogResult, ...]
    food_totals: NutritionTotalsValue
    workout_logs: tuple[WorkoutLogResult, ...]


class ProteinTargetResult(_ToolValue):
    """Separate stored recommended and explicitly active protein targets."""

    recommended_protein_target_g: int | None
    active_protein_target_g: int | None


def _nutrition_totals_result(
    totals: member_state.NutritionTotals,
) -> NutritionTotalsValue:
    return NutritionTotalsValue(
        calories=totals.calories,
        protein_g=totals.protein_g,
        carbs_g=totals.carbs_g,
        fat_g=totals.fat_g,
    )


def _stored_nutrition_totals_result(totals: dict[str, Any]) -> NutritionTotalsValue:
    return NutritionTotalsValue.model_validate(totals)


def _bodyweight_result(bodyweight: BodyweightLog) -> BodyweightResult:
    return BodyweightResult(
        local_date=bodyweight.log_date,
        weight_kg=bodyweight.weight_kg,
        recorded_at=bodyweight.recorded_at,
        note=bodyweight.note,
    )


def _optional_bodyweight_result(
    bodyweight: BodyweightLog | None,
) -> BodyweightResult | None:
    return None if bodyweight is None else _bodyweight_result(bodyweight)


def _food_log_result(food_log: FoodLog) -> FoodLogResult:
    return FoodLogResult(
        local_date=food_log.log_date,
        meal_type=food_log.meal_type,
        items=deepcopy(food_log.items),
        totals=_stored_nutrition_totals_result(food_log.totals),
        recorded_at=food_log.recorded_at,
        note=food_log.note,
    )


def _workout_log_result(workout_log: WorkoutLog) -> WorkoutLogResult:
    return WorkoutLogResult(
        local_date=workout_log.log_date,
        exercises=deepcopy(workout_log.exercises),
        recorded_at=workout_log.recorded_at,
        note=workout_log.note,
    )


def _day_state_result(state: member_state.DayState) -> DayStateResult:
    return DayStateResult(
        local_date=state.local_date,
        bodyweight=(
            None
            if state.bodyweight_log is None
            else _bodyweight_result(state.bodyweight_log)
        ),
        food_logs=tuple(_food_log_result(log) for log in state.food_logs),
        food_totals=_nutrition_totals_result(state.food_totals),
        workout_logs=tuple(_workout_log_result(log) for log in state.workout_logs),
    )


def _protein_target_result(
    state: protein_targets.ProteinTargetState,
) -> ProteinTargetResult:
    return ProteinTargetResult(
        recommended_protein_target_g=state.recommended_protein_target_g,
        active_protein_target_g=state.active_protein_target_g,
    )


def _optional_protein_target_result(
    state: protein_targets.ProteinTargetState | None,
) -> ProteinTargetResult | None:
    return None if state is None else _protein_target_result(state)


class BarbarikAgentTools:
    """Safe explicit operations bound to one trusted member request context.

    The backend, not the model, supplies a session factory, member identity, and
    aware event timestamp. Public writes never accept a timezone, local date,
    or timestamp, so the existing services remain authoritative for chronology.
    Every call owns a fresh session; mutations commit or roll back atomically.
    """

    __slots__ = ("_event_time", "_member_id", "_session_factory")

    def __init__(
        self,
        session_factory: Callable[[], Session],
        member_id: UUID,
        event_time: datetime,
    ) -> None:
        if not callable(session_factory):
            raise ValueError("session_factory must be callable")
        if not isinstance(member_id, UUID):
            raise ValueError("member_id must be a UUID")
        if (
            not isinstance(event_time, datetime)
            or event_time.tzinfo is None
            or event_time.utcoffset() is None
        ):
            raise ValueError("event_time must be a timezone-aware datetime")

        self._session_factory = session_factory
        self._member_id = member_id
        self._event_time = event_time

    def _read(self, operation: Callable[[Session], _ResultT]) -> _ResultT:
        """Run one read inside a fresh, short-lived session."""

        with self._session_factory() as session:
            return operation(session)

    def _commit_mutation(
        self,
        operation: Callable[[Session], _ResultT],
    ) -> _ResultT:
        """Commit one tool mutation, rolling its transaction back on failure."""

        with self._session_factory() as session:
            try:
                result = operation(session)
                session.commit()
            except Exception:
                session.rollback()
                raise
            return result

    def get_today_state(self) -> DayStateResult:
        """Return state for the trusted event instant in the member's timezone."""

        return self._read(
            lambda session: _day_state_result(
                member_state.get_today_state(
                    session,
                    self._member_id,
                    now=self._event_time,
                )
            )
        )

    def get_yesterday_state(self) -> DayStateResult:
        """Return the prior local day's persisted state for the bound member."""

        return self._read(
            lambda session: _day_state_result(
                member_state.get_yesterday_state(
                    session,
                    self._member_id,
                    now=self._event_time,
                )
            )
        )

    @validate_call
    def get_day_state(self, day: date) -> DayStateResult:
        """Return the bound member's state for one requested local date."""

        return self._read(
            lambda session: _day_state_result(
                member_state.get_day_state(session, self._member_id, day)
            )
        )

    @validate_call
    def get_bodyweight_for_day(self, day: date) -> BodyweightResult | None:
        """Return the bound member's bodyweight for one local date."""

        return self._read(
            lambda session: _optional_bodyweight_result(
                member_state.get_bodyweight_for_day(
                    session,
                    self._member_id,
                    day,
                )
            )
        )

    def get_latest_bodyweight(self) -> BodyweightResult | None:
        """Return the bound member's latest persisted bodyweight."""

        return self._read(
            lambda session: _optional_bodyweight_result(
                member_state.get_latest_bodyweight(session, self._member_id)
            )
        )

    @validate_call
    def log_bodyweight(self, weight_kg: Decimal) -> BodyweightResult:
        """Persist bodyweight at the trusted event timestamp."""

        if not isinstance(weight_kg, Decimal):
            raise ValueError("weight_kg must be a Decimal")

        return self._commit_mutation(
            lambda session: _bodyweight_result(
                member_state.log_bodyweight(
                    session,
                    self._member_id,
                    weight_kg,
                    self._event_time,
                )
            )
        )

    @validate_call
    def get_food_for_day(self, day: date) -> tuple[FoodLogResult, ...]:
        """Return the bound member's stored meals for one local date."""

        return self._read(
            lambda session: tuple(
                _food_log_result(log)
                for log in member_state.get_food_for_day(
                    session,
                    self._member_id,
                    day,
                )
            )
        )

    @validate_call
    def get_food_totals_for_day(self, day: date) -> NutritionTotalsValue:
        """Return authoritative meal totals for one requested local date."""

        return self._read(
            lambda session: _nutrition_totals_result(
                member_state.get_food_totals_for_day(
                    session,
                    self._member_id,
                    day,
                )
            )
        )

    def get_today_food_totals(self) -> NutritionTotalsValue:
        """Return authoritative totals for the trusted member-local today."""

        return self._read(
            lambda session: _nutrition_totals_result(
                member_state.get_today_food_totals(
                    session,
                    self._member_id,
                    now=self._event_time,
                )
            )
        )

    @validate_call
    def log_food(
        self,
        items: list[dict[str, Any]],
        totals: NutritionTotalsValue,
        *,
        meal_type: str | None = None,
        note: str | None = None,
    ) -> FoodLogResult:
        """Persist one structured model-estimated meal at the trusted instant."""

        service_totals = member_state.NutritionTotals(
            calories=totals.calories,
            protein_g=totals.protein_g,
            carbs_g=totals.carbs_g,
            fat_g=totals.fat_g,
        )
        return self._commit_mutation(
            lambda session: _food_log_result(
                member_state.log_food(
                    session,
                    self._member_id,
                    items,
                    service_totals,
                    self._event_time,
                    meal_type=meal_type,
                    note=note,
                )
            )
        )

    @validate_call
    def get_workouts_for_day(self, day: date) -> tuple[WorkoutLogResult, ...]:
        """Return the bound member's workouts for one local date."""

        return self._read(
            lambda session: tuple(
                _workout_log_result(log)
                for log in member_state.get_workouts_for_day(
                    session,
                    self._member_id,
                    day,
                )
            )
        )

    def get_recent_workouts(self) -> tuple[WorkoutLogResult, ...]:
        """Return a fixed bounded set of the bound member's recent workouts."""

        return self._read(
            lambda session: tuple(
                _workout_log_result(log)
                for log in member_state.get_recent_workouts(
                    session,
                    self._member_id,
                    limit=_HISTORY_LIMIT,
                )
            )
        )

    @validate_call
    def get_exercise_history(
        self,
        exercise_id: UUID,
    ) -> tuple[WorkoutLogResult, ...]:
        """Return fixed bounded history for one canonical exercise ID."""

        return self._read(
            lambda session: tuple(
                _workout_log_result(log)
                for log in member_state.get_exercise_history(
                    session,
                    self._member_id,
                    exercise_id,
                    limit=_HISTORY_LIMIT,
                )
            )
        )

    @validate_call
    def log_workout(
        self,
        exercises: tuple[WorkoutExerciseValue, ...],
        *,
        note: str | None = None,
    ) -> WorkoutLogResult:
        """Persist a completed structured workout at the trusted instant."""

        service_exercises = tuple(
            member_state.WorkoutExerciseInput(
                exercise_id=exercise.exercise_id,
                sets=tuple(
                    member_state.WorkoutSetInput(
                        reps=completed_set.reps,
                        weight_kg=completed_set.weight_kg,
                    )
                    for completed_set in exercise.sets
                ),
            )
            for exercise in exercises
        )
        return self._commit_mutation(
            lambda session: _workout_log_result(
                member_state.log_workout(
                    session,
                    self._member_id,
                    service_exercises,
                    self._event_time,
                    note=note,
                )
            )
        )

    def get_protein_target_state(self) -> ProteinTargetResult | None:
        """Return recommendation and active target without fabricating either."""

        return self._read(
            lambda session: _optional_protein_target_result(
                protein_targets.get_protein_target_state(
                    session,
                    self._member_id,
                )
            )
        )

    def calculate_and_store_protein_recommendation(self) -> ProteinTargetResult:
        """Store the deterministic recommendation without activating it."""

        return self._commit_mutation(
            lambda session: _protein_target_result(
                protein_targets.calculate_and_store_protein_recommendation(
                    session,
                    self._member_id,
                )
            )
        )

    def accept_protein_recommendation(self) -> ProteinTargetResult:
        """Explicitly make the stored recommendation active."""

        return self._commit_mutation(
            lambda session: _protein_target_result(
                protein_targets.accept_protein_recommendation(
                    session,
                    self._member_id,
                )
            )
        )

    @validate_call
    def override_protein_target(self, target_g: StrictInt) -> ProteinTargetResult:
        """Explicitly set the active target without replacing the recommendation."""

        return self._commit_mutation(
            lambda session: _protein_target_result(
                protein_targets.override_protein_target(
                    session,
                    self._member_id,
                    target_g,
                )
            )
        )
