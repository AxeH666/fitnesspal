"""Tests for the explicit member-bound agent tool surface."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
from decimal import Decimal
import inspect
import json
import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import Engine, Table, create_engine, func, select
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.sql.compiler import TypeCompiler

from app.agent_tools import (
    BarbarikAgentTools,
    DayStateResult,
    NutritionTotalsValue,
    WorkoutExerciseValue,
    WorkoutSetValue,
)
from app.db.base import Base
from app.models import (
    BodyweightLog,
    Exercise,
    FoodLog,
    Member,
    MemberNutritionTarget,
    MemberProfile,
    WorkoutLog,
    WorkoutPlan,
)
from app.services import member_state, protein_targets


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(
    _type: JSONB,
    _compiler: TypeCompiler,
    **_kwargs: object,
) -> str:
    return "JSON"


@compiles(ARRAY, "sqlite")
def _compile_postgres_array_for_sqlite(
    _type: object,
    _compiler: TypeCompiler,
    **_kwargs: object,
) -> str:
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_postgres_uuid_for_sqlite(
    _type: object,
    _compiler: TypeCompiler,
    **_kwargs: object,
) -> str:
    return "CHAR(32)"


class AgentToolContractTests(unittest.TestCase):
    """Verify exact delegation and the safe public boundary."""

    member_id: UUID
    event_time: datetime
    requested_day: date
    session_mock: MagicMock
    session: Session
    factory_mock: MagicMock
    factory: Callable[[], Session]
    tools: BarbarikAgentTools

    def setUp(self) -> None:
        self.member_id = uuid4()
        self.event_time = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)
        self.requested_day = date(2026, 8, 31)
        self.session_mock = MagicMock(spec=Session)
        self.session = cast(Session, self.session_mock)
        self.session_mock.__enter__.return_value = self.session
        self.factory_mock = MagicMock(return_value=self.session_mock)
        self.factory = cast(Callable[[], Session], self.factory_mock)
        self.tools = BarbarikAgentTools(
            self.factory,
            self.member_id,
            self.event_time,
        )

    def _bodyweight(self) -> BodyweightLog:
        return BodyweightLog(
            member_id=self.member_id,
            log_date=self.requested_day,
            weight_kg=Decimal("79.40"),
            recorded_at=self.event_time,
            note="morning",
        )

    def _food(self) -> FoodLog:
        return FoodLog(
            member_id=self.member_id,
            log_date=self.requested_day,
            meal_type="breakfast",
            items=[{"name": "Eggs", "qty": 4, "unit": "eggs"}],
            totals={
                "calories": 280,
                "protein_g": 24,
                "carbs_g": 2,
                "fat_g": 20,
            },
            recorded_at=self.event_time,
            note="estimated",
        )

    def _workout(self) -> WorkoutLog:
        return WorkoutLog(
            member_id=self.member_id,
            log_date=self.requested_day,
            exercises=[
                {
                    "exercise_id": str(uuid4()),
                    "name": "Barbell Bench Press",
                    "sets": [{"reps": 8, "weight_kg": 80}],
                }
            ],
            recorded_at=self.event_time,
            note="upper body",
        )

    def _state(self) -> member_state.DayState:
        return member_state.DayState(
            self.member_id,
            self.requested_day,
            self._bodyweight(),
            (self._food(),),
            member_state.NutritionTotals(280, 24, 2, 20),
            (self._workout(),),
        )

    def _assert_delegates(
        self,
        target: str,
        service_result: object,
        invoke: Callable[[], object],
        expected_args: tuple[object, ...],
        expected_kwargs: dict[str, object] | None = None,
        *,
        mutation: bool = False,
    ) -> None:
        self.session_mock.reset_mock()
        self.factory_mock.reset_mock()
        with patch(target, return_value=service_result) as operation:
            invoke()
        operation.assert_called_once_with(
            *expected_args,
            **(expected_kwargs or {}),
        )
        self.factory_mock.assert_called_once_with()
        self.session_mock.__enter__.assert_called_once_with()
        self.session_mock.__exit__.assert_called_once()
        if mutation:
            self.session_mock.commit.assert_called_once_with()
        else:
            self.session_mock.commit.assert_not_called()
        self.session_mock.rollback.assert_not_called()

    def test_public_surface_is_explicit_and_hides_trusted_context(self) -> None:
        expected = {
            "accept_protein_recommendation",
            "calculate_and_store_protein_recommendation",
            "get_bodyweight_for_day",
            "get_day_state",
            "get_exercise_history",
            "get_food_for_day",
            "get_food_totals_for_day",
            "get_latest_bodyweight",
            "get_protein_target_state",
            "get_recent_workouts",
            "get_today_food_totals",
            "get_today_state",
            "get_yesterday_state",
            "get_workouts_for_day",
            "log_bodyweight",
            "log_food",
            "log_workout",
            "override_protein_target",
        }
        methods = {
            name: method
            for name, method in inspect.getmembers(
                BarbarikAgentTools,
                predicate=inspect.isfunction,
            )
            if not name.startswith("_")
        }
        self.assertEqual(set(methods), expected)
        forbidden = (
            "session",
            "member_id",
            "event_time",
            "timezone",
            "log_date",
            "sql",
            "query",
        )
        for name, method in methods.items():
            with self.subTest(method=name):
                signature = str(inspect.signature(method)).lower()
                for fragment in forbidden:
                    self.assertNotIn(fragment, signature)

    def test_every_public_tool_delegates_to_its_exact_service(self) -> None:
        state = self._state()
        bodyweight = self._bodyweight()
        food = self._food()
        workout = self._workout()
        totals = member_state.NutritionTotals(280, 24, 2, 20)
        target_state = protein_targets.ProteinTargetState(self.member_id, 158, 140)
        exercise_id = uuid4()
        items = [{"name": "Eggs", "qty": 4, "unit": "eggs"}]
        tool_totals = NutritionTotalsValue(
            calories=280,
            protein_g=24,
            carbs_g=2,
            fat_g=20,
        )
        tool_exercises = (
            WorkoutExerciseValue(
                exercise_id=exercise_id,
                sets=(WorkoutSetValue(reps=8, weight_kg=80),),
            ),
        )
        service_exercises = (
            member_state.WorkoutExerciseInput(
                exercise_id,
                (member_state.WorkoutSetInput(8, 80),),
            ),
        )
        cases: tuple[
            tuple[
                str,
                object,
                Callable[[], object],
                tuple[object, ...],
                dict[str, object],
                bool,
            ],
            ...,
        ] = (
            (
                "app.agent_tools.member_state.get_today_state",
                state,
                self.tools.get_today_state,
                (self.session, self.member_id),
                {"now": self.event_time},
                False,
            ),
            (
                "app.agent_tools.member_state.get_yesterday_state",
                state,
                self.tools.get_yesterday_state,
                (self.session, self.member_id),
                {"now": self.event_time},
                False,
            ),
            (
                "app.agent_tools.member_state.get_day_state",
                state,
                lambda: self.tools.get_day_state(self.requested_day),
                (self.session, self.member_id, self.requested_day),
                {},
                False,
            ),
            (
                "app.agent_tools.member_state.get_bodyweight_for_day",
                bodyweight,
                lambda: self.tools.get_bodyweight_for_day(self.requested_day),
                (self.session, self.member_id, self.requested_day),
                {},
                False,
            ),
            (
                "app.agent_tools.member_state.get_latest_bodyweight",
                bodyweight,
                self.tools.get_latest_bodyweight,
                (self.session, self.member_id),
                {},
                False,
            ),
            (
                "app.agent_tools.member_state.log_bodyweight",
                bodyweight,
                lambda: self.tools.log_bodyweight(Decimal("79.40")),
                (self.session, self.member_id, Decimal("79.40"), self.event_time),
                {},
                True,
            ),
            (
                "app.agent_tools.member_state.get_food_for_day",
                (food,),
                lambda: self.tools.get_food_for_day(self.requested_day),
                (self.session, self.member_id, self.requested_day),
                {},
                False,
            ),
            (
                "app.agent_tools.member_state.get_food_totals_for_day",
                totals,
                lambda: self.tools.get_food_totals_for_day(self.requested_day),
                (self.session, self.member_id, self.requested_day),
                {},
                False,
            ),
            (
                "app.agent_tools.member_state.get_today_food_totals",
                totals,
                self.tools.get_today_food_totals,
                (self.session, self.member_id),
                {"now": self.event_time},
                False,
            ),
            (
                "app.agent_tools.member_state.log_food",
                food,
                lambda: self.tools.log_food(
                    items,
                    tool_totals,
                    meal_type="breakfast",
                    note="estimated",
                ),
                (self.session, self.member_id, items, totals, self.event_time),
                {"meal_type": "breakfast", "note": "estimated"},
                True,
            ),
            (
                "app.agent_tools.member_state.get_workouts_for_day",
                (workout,),
                lambda: self.tools.get_workouts_for_day(self.requested_day),
                (self.session, self.member_id, self.requested_day),
                {},
                False,
            ),
            (
                "app.agent_tools.member_state.get_recent_workouts",
                (workout,),
                self.tools.get_recent_workouts,
                (self.session, self.member_id),
                {"limit": 10},
                False,
            ),
            (
                "app.agent_tools.member_state.get_exercise_history",
                (workout,),
                lambda: self.tools.get_exercise_history(exercise_id),
                (self.session, self.member_id, exercise_id),
                {"limit": 10},
                False,
            ),
            (
                "app.agent_tools.member_state.log_workout",
                workout,
                lambda: self.tools.log_workout(tool_exercises, note="upper body"),
                (self.session, self.member_id, service_exercises, self.event_time),
                {"note": "upper body"},
                True,
            ),
            (
                "app.agent_tools.protein_targets.get_protein_target_state",
                target_state,
                self.tools.get_protein_target_state,
                (self.session, self.member_id),
                {},
                False,
            ),
            (
                "app.agent_tools.protein_targets."
                "calculate_and_store_protein_recommendation",
                target_state,
                self.tools.calculate_and_store_protein_recommendation,
                (self.session, self.member_id),
                {},
                True,
            ),
            (
                "app.agent_tools.protein_targets.accept_protein_recommendation",
                target_state,
                self.tools.accept_protein_recommendation,
                (self.session, self.member_id),
                {},
                True,
            ),
            (
                "app.agent_tools.protein_targets.override_protein_target",
                target_state,
                lambda: self.tools.override_protein_target(140),
                (self.session, self.member_id, 140),
                {},
                True,
            ),
        )
        for target, result, invoke, arguments, keywords, mutation in cases:
            with self.subTest(service=target):
                self._assert_delegates(
                    target,
                    result,
                    invoke,
                    arguments,
                    keywords,
                    mutation=mutation,
                )

    def test_failed_write_rolls_back_and_surfaces_the_error(self) -> None:
        with patch(
            "app.agent_tools.member_state.log_bodyweight",
            side_effect=ValueError("invalid persisted input"),
        ) as operation:
            with self.assertRaisesRegex(ValueError, "invalid persisted input"):
                self.tools.log_bodyweight(Decimal("0"))
        operation.assert_called_once_with(
            self.session,
            self.member_id,
            Decimal("0"),
            self.event_time,
        )
        self.factory_mock.assert_called_once_with()
        self.session_mock.commit.assert_not_called()
        self.session_mock.rollback.assert_called_once_with()
        self.session_mock.__exit__.assert_called_once()

    def test_json_shaped_arguments_are_coerced_before_delegation(self) -> None:
        self._assert_delegates(
            "app.agent_tools.member_state.get_day_state",
            self._state(),
            lambda: self.tools.get_day_state(cast(date, "2026-08-31")),
            (self.session, self.member_id, self.requested_day),
        )
        self._assert_delegates(
            "app.agent_tools.member_state.log_bodyweight",
            self._bodyweight(),
            lambda: self.tools.log_bodyweight(cast(Decimal, "79.40")),
            (self.session, self.member_id, Decimal("79.40"), self.event_time),
            mutation=True,
        )
        raw_totals = cast(
            NutritionTotalsValue,
            {
                "calories": 280,
                "protein_g": 24,
                "carbs_g": 2,
                "fat_g": 20,
            },
        )
        items = [{"name": "Eggs", "qty": 4, "unit": "eggs"}]
        self._assert_delegates(
            "app.agent_tools.member_state.log_food",
            self._food(),
            lambda: self.tools.log_food(items, raw_totals),
            (
                self.session,
                self.member_id,
                items,
                member_state.NutritionTotals(280, 24, 2, 20),
                self.event_time,
            ),
            {"meal_type": None, "note": None},
            mutation=True,
        )
        exercise_id = uuid4()
        raw_exercises = cast(
            tuple[WorkoutExerciseValue, ...],
            [
                {
                    "exercise_id": str(exercise_id),
                    "sets": [{"reps": 8, "weight_kg": 80}],
                }
            ],
        )
        expected = (
            member_state.WorkoutExerciseInput(
                exercise_id,
                (member_state.WorkoutSetInput(8, 80),),
            ),
        )
        self._assert_delegates(
            "app.agent_tools.member_state.log_workout",
            self._workout(),
            lambda: self.tools.log_workout(raw_exercises),
            (self.session, self.member_id, expected, self.event_time),
            {"note": None},
            mutation=True,
        )


class AgentToolPersistenceTests(unittest.TestCase):
    """Exercise persistence, member isolation, and chronology through tools."""

    engine: Engine
    setup_session: Session

    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        with self.engine.connect() as connection:
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        Base.metadata.create_all(
            self.engine,
            tables=[
                cast(Table, Member.__table__),
                cast(Table, MemberProfile.__table__),
                cast(Table, MemberNutritionTarget.__table__),
                cast(Table, BodyweightLog.__table__),
                cast(Table, FoodLog.__table__),
                cast(Table, Exercise.__table__),
                cast(Table, WorkoutPlan.__table__),
                cast(Table, WorkoutLog.__table__),
            ],
        )
        self.setup_session = self._new_session()

    def tearDown(self) -> None:
        self.setup_session.close()
        self.engine.dispose()

    def _new_session(self) -> Session:
        return Session(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )

    def _add_member(self, email: str, timezone_name: str) -> Member:
        member = Member(email=email, password_hash="test-only")
        self.setup_session.add(member)
        self.setup_session.flush()
        self.setup_session.add(
            MemberProfile(
                member_id=member.id,
                goal="fat_loss",
                activity_level="moderate",
                training_experience="beginner",
                timezone=timezone_name,
            )
        )
        self.setup_session.flush()
        return member

    def _add_exercise(self, name: str) -> Exercise:
        exercise = Exercise(name=name)
        self.setup_session.add(exercise)
        self.setup_session.flush()
        return exercise

    def _workout_input(
        self,
        exercise_id: UUID,
        reps: int,
        weight_kg: int,
    ) -> tuple[WorkoutExerciseValue, ...]:
        return (
            WorkoutExerciseValue(
                exercise_id=exercise_id,
                sets=(WorkoutSetValue(reps=reps, weight_kg=weight_kg),),
            ),
        )

    def _log_day(
        self,
        tools: BarbarikAgentTools,
        exercise_id: UUID,
        marker: str,
        weight: str,
        totals: tuple[int, int, int, int],
    ) -> None:
        tools.log_bodyweight(Decimal(weight))
        tools.log_food(
            [{"name": marker, "qty": 1, "unit": "serving"}],
            NutritionTotalsValue(
                calories=totals[0],
                protein_g=totals[1],
                carbs_g=totals[2],
                fat_g=totals[3],
            ),
            note=f"{marker}-meal",
        )
        tools.log_workout(
            self._workout_input(exercise_id, 8, 80),
            note=f"{marker}-workout",
        )

    def _assert_state(
        self,
        state: DayStateResult,
        expected_day: date,
        expected_weight: str,
        expected_calories: int,
        marker: str,
    ) -> None:
        self.assertEqual(state.local_date, expected_day)
        self.assertIsNotNone(state.bodyweight)
        assert state.bodyweight is not None
        self.assertEqual(state.bodyweight.weight_kg, Decimal(expected_weight))
        self.assertEqual(state.food_totals.calories, expected_calories)
        self.assertEqual(
            [workout.note for workout in state.workout_logs],
            [f"{marker}-workout"],
        )

    def _count(self, table: Table, member_id: UUID) -> int:
        with self._new_session() as session:
            count = session.scalar(
                select(func.count())
                .select_from(table)
                .where(table.c.member_id == member_id)
            )
        assert count is not None
        return count

    def test_persistence_member_scope_and_local_midnight_isolation(self) -> None:
        india = self._add_member("india@example.test", "Asia/Kolkata")
        new_york = self._add_member("new-york@example.test", "America/New_York")
        bench = self._add_exercise("Barbell Bench Press")
        squat = self._add_exercise("Barbell Back Squat")
        self.setup_session.commit()
        india_id, new_york_id = india.id, new_york.id
        bench_id, squat_id = bench.id, squat.id
        before_midnight = datetime(2026, 8, 30, 18, 29, 59, tzinfo=timezone.utc)
        at_midnight = datetime(2026, 8, 30, 18, 30, tzinfo=timezone.utc)

        india_a = BarbarikAgentTools(self._new_session, india_id, before_midnight)
        india_b = BarbarikAgentTools(self._new_session, india_id, at_midnight)
        new_york_same_time = BarbarikAgentTools(
            self._new_session,
            new_york_id,
            at_midnight,
        )
        self._log_day(india_a, bench_id, "india-a", "80.00", (400, 30, 50, 10))
        self._log_day(india_b, bench_id, "india-b", "79.80", (500, 40, 60, 15))
        self._log_day(
            new_york_same_time,
            squat_id,
            "new-york",
            "90.00",
            (900, 70, 100, 30),
        )
        self.setup_session.close()

        india_reader = BarbarikAgentTools(self._new_session, india_id, at_midnight)
        new_york_reader = BarbarikAgentTools(
            self._new_session,
            new_york_id,
            at_midnight,
        )
        india_today = india_reader.get_today_state()
        self._assert_state(
            india_today,
            date(2026, 8, 31),
            "79.80",
            500,
            "india-b",
        )
        self.assertEqual(
            json.loads(india_today.model_dump_json())["local_date"],
            "2026-08-31",
        )
        self._assert_state(
            india_reader.get_day_state(date(2026, 8, 30)),
            date(2026, 8, 30),
            "80.00",
            400,
            "india-a",
        )
        self._assert_state(
            india_reader.get_yesterday_state(),
            date(2026, 8, 30),
            "80.00",
            400,
            "india-a",
        )
        self._assert_state(
            new_york_reader.get_today_state(),
            date(2026, 8, 30),
            "90.00",
            900,
            "new-york",
        )
        new_york_yesterday = new_york_reader.get_yesterday_state()
        self.assertEqual(new_york_yesterday.local_date, date(2026, 8, 29))
        self.assertIsNone(new_york_yesterday.bodyweight)
        self.assertEqual(new_york_yesterday.food_logs, ())
        self.assertEqual(new_york_yesterday.food_totals.calories, 0)
        self.assertEqual(new_york_yesterday.workout_logs, ())
        self.assertEqual(india_reader.get_today_food_totals().protein_g, 40)
        self.assertEqual(
            india_reader.get_food_totals_for_day(date(2026, 8, 30)).calories,
            400,
        )
        self.assertEqual(
            [meal.note for meal in india_reader.get_food_for_day(date(2026, 8, 31))],
            ["india-b-meal"],
        )
        expected_india_history = ["india-b-workout", "india-a-workout"]
        self.assertEqual(
            [workout.note for workout in india_reader.get_recent_workouts()],
            expected_india_history,
        )
        self.assertEqual(
            [workout.note for workout in india_reader.get_exercise_history(bench_id)],
            expected_india_history,
        )
        self.assertEqual(new_york_reader.get_exercise_history(bench_id), ())
        self.assertEqual(
            [
                workout.note
                for workout in new_york_reader.get_workouts_for_day(
                    date(2026, 8, 30)
                )
            ],
            ["new-york-workout"],
        )
        latest = india_reader.get_latest_bodyweight()
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.weight_kg, Decimal("79.80"))
        self.assertIsNone(india_reader.get_bodyweight_for_day(date(2026, 8, 29)))

    def test_protein_lifecycle_persists(self) -> None:
        member = self._add_member("protein@example.test", "Asia/Kolkata")
        self.setup_session.commit()
        member_id = member.id
        first_time = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
        second_time = datetime(2026, 9, 2, 8, 0, tzinfo=timezone.utc)
        first = BarbarikAgentTools(self._new_session, member_id, first_time)
        first.log_bodyweight(Decimal("79.00"))
        recommended = first.calculate_and_store_protein_recommendation()
        self.assertEqual(
            (
                recommended.recommended_protein_target_g,
                recommended.active_protein_target_g,
            ),
            (158, None),
        )
        self.assertEqual(first.accept_protein_recommendation().active_protein_target_g, 158)

        second = BarbarikAgentTools(self._new_session, member_id, second_time)
        second.log_bodyweight(Decimal("80.00"))
        recalculated = second.calculate_and_store_protein_recommendation()
        self.assertEqual(
            (
                recalculated.recommended_protein_target_g,
                recalculated.active_protein_target_g,
            ),
            (160, 158),
        )
        overridden = second.override_protein_target(140)
        self.assertEqual(
            BarbarikAgentTools(
                self._new_session,
                member_id,
                second_time,
            ).get_protein_target_state(),
            overridden,
        )

    def test_invalid_mutations_create_no_rows_and_recover(self) -> None:
        member = self._add_member("invalid@example.test", "Asia/Kolkata")
        exercise = self._add_exercise("Barbell Bench Press")
        self.setup_session.commit()
        member_id, exercise_id = member.id, exercise.id
        tools = BarbarikAgentTools(
            self._new_session,
            member_id,
            datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc),
        )
        totals = NutritionTotalsValue(
            calories=100,
            protein_g=10,
            carbs_g=10,
            fat_g=2,
        )

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            tools.log_bodyweight(Decimal("0"))
        with self.assertRaisesRegex(ValueError, "at least one food item"):
            tools.log_food([], totals)
        with self.assertRaisesRegex(ValueError, "at least one exercise"):
            tools.log_workout(())
        with self.assertRaisesRegex(LookupError, "Unknown exercise IDs"):
            tools.log_workout(self._workout_input(uuid4(), 8, 80))
        with self.assertRaisesRegex(LookupError, "No protein recommendation"):
            tools.accept_protein_recommendation()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            tools.override_protein_target(0)
        with self.assertRaises(ValidationError):
            tools.override_protein_target(cast(int, "140"))

        self.assertEqual(self._count(cast(Table, BodyweightLog.__table__), member_id), 0)
        self.assertEqual(self._count(cast(Table, FoodLog.__table__), member_id), 0)
        self.assertEqual(self._count(cast(Table, WorkoutLog.__table__), member_id), 0)
        self.assertEqual(
            self._count(cast(Table, MemberNutritionTarget.__table__), member_id),
            0,
        )
        valid = tools.log_workout(self._workout_input(exercise_id, 8, 80))
        self.assertEqual(valid.local_date, date(2026, 9, 1))
        self.assertEqual(self._count(cast(Table, WorkoutLog.__table__), member_id), 1)


if __name__ == "__main__":
    unittest.main()
