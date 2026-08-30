"""Tests for date-aware workout logging and member-scoped history."""

from __future__ import annotations

from datetime import date, datetime, timezone
import math
import unittest
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import Engine, Table, create_engine, func, select
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.sql.compiler import TypeCompiler

from app.db.base import Base
from app.models import (
    BodyweightLog,
    Exercise,
    FoodLog,
    Member,
    MemberProfile,
    WorkoutLog,
    WorkoutPlan,
)
from app.services.member_state import (
    WorkoutExerciseInput,
    WorkoutSetInput,
    get_day_state,
    get_exercise_history,
    get_recent_workouts,
    get_today_state,
    get_workouts_for_day,
    log_workout,
)


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(
    _type: JSONB,
    _compiler: TypeCompiler,
    **_kwargs: object,
) -> str:
    """Allow production JSONB columns in this focused SQLite fixture."""

    return "JSON"


@compiles(ARRAY, "sqlite")
def _compile_postgres_array_for_sqlite(
    _type: object,
    _compiler: TypeCompiler,
    **_kwargs: object,
) -> str:
    """Allow unused exercise-library array columns in the SQLite fixture."""

    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_postgres_uuid_for_sqlite(
    _type: object,
    _compiler: TypeCompiler,
    **_kwargs: object,
) -> str:
    """Allow production UUID columns in this focused SQLite fixture."""

    return "CHAR(32)"


class WorkoutLoggingTests(unittest.TestCase):
    """Exercise structured workout operations without shared infrastructure."""

    engine: Engine
    session: Session

    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(
            self.engine,
            tables=[
                cast(Table, Member.__table__),
                cast(Table, MemberProfile.__table__),
                cast(Table, BodyweightLog.__table__),
                cast(Table, FoodLog.__table__),
                cast(Table, Exercise.__table__),
                cast(Table, WorkoutPlan.__table__),
                cast(Table, WorkoutLog.__table__),
            ],
        )
        self.session = Session(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()

    def _add_member(
        self,
        email: str,
        timezone_name: str = "Asia/Kolkata",
    ) -> Member:
        member = Member(email=email, password_hash="test-only")
        self.session.add(member)
        self.session.flush()
        self.session.add(
            MemberProfile(
                member_id=member.id,
                goal="general_fitness",
                activity_level="moderate",
                training_experience="beginner",
                timezone=timezone_name,
            )
        )
        self.session.flush()
        return member

    def _add_exercise(self, name: str) -> Exercise:
        exercise = Exercise(name=name)
        self.session.add(exercise)
        self.session.flush()
        return exercise

    def _exercise_input(
        self,
        exercise: Exercise,
        *sets: tuple[int, int | float],
    ) -> WorkoutExerciseInput:
        return WorkoutExerciseInput(
            exercise_id=exercise.id,
            sets=tuple(
                WorkoutSetInput(reps=reps, weight_kg=weight_kg)
                for reps, weight_kg in sets
            ),
        )

    def _workout_count(self, member_id: UUID) -> int:
        count = self.session.scalar(
            select(func.count())
            .select_from(WorkoutLog)
            .where(WorkoutLog.member_id == member_id)
        )
        assert count is not None
        return count

    def assert_recorded_at_matches(
        self,
        workout_log: WorkoutLog,
        expected: datetime,
    ) -> None:
        actual = workout_log.recorded_at
        if actual.tzinfo is None:
            actual = actual.replace(tzinfo=timezone.utc)
        self.assertEqual(actual, expected.astimezone(timezone.utc))

    def test_workout_log_persists_details_timestamp_and_local_date(self) -> None:
        member = self._add_member("workout@example.test")
        bench = self._add_exercise("Barbell Bench Press")
        pull_up = self._add_exercise("Pull Up")
        event_time = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)
        mutable_sets = [
            WorkoutSetInput(reps=8, weight_kg=80),
            WorkoutSetInput(reps=6, weight_kg=80),
        ]
        exercises = (
            WorkoutExerciseInput(
                exercise_id=bench.id,
                sets=cast(tuple[WorkoutSetInput, ...], mutable_sets),
            ),
            self._exercise_input(pull_up, (10, 0)),
        )

        workout = log_workout(
            self.session,
            member.id,
            exercises,
            event_time,
            note="Upper body",
        )

        self.assertEqual(workout.member_id, member.id)
        self.assertEqual(workout.log_date, date(2026, 8, 31))
        self.assertEqual(workout.note, "Upper body")
        self.assertEqual(
            workout.exercises,
            [
                {
                    "exercise_id": str(bench.id),
                    "name": "Barbell Bench Press",
                    "sets": [
                        {"reps": 8, "weight_kg": 80},
                        {"reps": 6, "weight_kg": 80},
                    ],
                },
                {
                    "exercise_id": str(pull_up.id),
                    "name": "Pull Up",
                    "sets": [{"reps": 10, "weight_kg": 0}],
                },
            ],
        )
        self.assert_recorded_at_matches(workout, event_time)
        self.assertEqual(self._workout_count(member.id), 1)

        mutable_sets[0] = WorkoutSetInput(reps=1, weight_kg=1)
        self.session.commit()
        with Session(self.engine) as persisted_session:
            persisted = get_workouts_for_day(
                persisted_session,
                member.id,
                date(2026, 8, 31),
            )
            self.assertEqual(len(persisted), 1)
            self.assertEqual(persisted[0].exercises[0]["sets"][0]["reps"], 8)
            self.assertEqual(persisted[0].exercises[0]["sets"][0]["weight_kg"], 80)

    def test_same_day_workouts_append_and_use_event_order(self) -> None:
        member = self._add_member("append@example.test")
        bench = self._add_exercise("Barbell Bench Press")
        later_time = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
        earlier_time = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)

        later = log_workout(
            self.session,
            member.id,
            (self._exercise_input(bench, (6, 85)),),
            later_time,
        )
        earlier = log_workout(
            self.session,
            member.id,
            (self._exercise_input(bench, (8, 80)),),
            earlier_time,
        )

        self.assertNotEqual(earlier.id, later.id)
        self.assertEqual(self._workout_count(member.id), 2)
        workouts = get_workouts_for_day(self.session, member.id, date(2026, 9, 1))
        self.assertEqual([workout.id for workout in workouts], [earlier.id, later.id])

    def test_local_midnight_separates_days_and_updates_today_state(self) -> None:
        member = self._add_member("midnight@example.test")
        squat = self._add_exercise("Barbell Back Squat")
        before_midnight = datetime(2026, 8, 30, 18, 29, 59, tzinfo=timezone.utc)
        at_midnight = datetime(2026, 8, 30, 18, 30, 0, tzinfo=timezone.utc)

        day_a_workout = log_workout(
            self.session,
            member.id,
            (self._exercise_input(squat, (8, 100)),),
            before_midnight,
        )
        day_a_before = get_day_state(self.session, member.id, date(2026, 8, 30))
        day_b_workout = log_workout(
            self.session,
            member.id,
            (self._exercise_input(squat, (6, 105)),),
            at_midnight,
        )

        day_a_after = get_day_state(self.session, member.id, date(2026, 8, 30))
        day_b = get_day_state(self.session, member.id, date(2026, 8, 31))
        today = get_today_state(self.session, member.id, now=at_midnight)

        self.assertEqual(
            [workout.id for workout in day_a_before.workout_logs],
            [day_a_workout.id],
        )
        self.assertEqual(
            [workout.id for workout in day_a_after.workout_logs],
            [day_a_workout.id],
        )
        self.assertEqual([workout.id for workout in day_b.workout_logs], [day_b_workout.id])
        self.assertEqual(today.local_date, date(2026, 8, 31))
        self.assertEqual([workout.id for workout in today.workout_logs], [day_b_workout.id])

    def test_same_instant_and_histories_remain_member_scoped(self) -> None:
        india_member = self._add_member("india@example.test", "Asia/Kolkata")
        new_york_member = self._add_member("new-york@example.test", "America/New_York")
        bench = self._add_exercise("Barbell Bench Press")
        event_time = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)

        india_workout = log_workout(
            self.session,
            india_member.id,
            (self._exercise_input(bench, (8, 80)),),
            event_time,
        )
        new_york_workout = log_workout(
            self.session,
            new_york_member.id,
            (self._exercise_input(bench, (5, 100)),),
            event_time,
        )

        self.assertEqual(india_workout.log_date, date(2026, 8, 31))
        self.assertEqual(new_york_workout.log_date, date(2026, 8, 30))
        self.assertEqual(
            get_workouts_for_day(self.session, india_member.id, date(2026, 8, 30)),
            (),
        )
        self.assertEqual(
            get_workouts_for_day(self.session, new_york_member.id, date(2026, 8, 31)),
            (),
        )
        self.assertEqual(
            [workout.id for workout in get_recent_workouts(self.session, india_member.id)],
            [india_workout.id],
        )
        self.assertEqual(
            [
                workout.id
                for workout in get_exercise_history(
                    self.session,
                    new_york_member.id,
                    bench.id,
                )
            ],
            [new_york_workout.id],
        )

    def test_recent_workouts_are_newest_first_and_limited(self) -> None:
        member = self._add_member("recent@example.test")
        deadlift = self._add_exercise("Conventional Deadlift")
        oldest = log_workout(
            self.session,
            member.id,
            (self._exercise_input(deadlift, (5, 120)),),
            datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc),
        )
        middle = log_workout(
            self.session,
            member.id,
            (self._exercise_input(deadlift, (4, 130)),),
            datetime(2026, 9, 2, 8, 0, tzinfo=timezone.utc),
        )
        newest = log_workout(
            self.session,
            member.id,
            (self._exercise_input(deadlift, (3, 140)),),
            datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc),
        )

        recent = get_recent_workouts(self.session, member.id, limit=2)

        self.assertEqual([workout.id for workout in recent], [newest.id, middle.id])
        self.assertNotIn(oldest.id, [workout.id for workout in recent])

    def test_exercise_history_filters_orders_and_limits(self) -> None:
        member = self._add_member("history@example.test")
        bench = self._add_exercise("Barbell Bench Press")
        squat = self._add_exercise("Barbell Back Squat")
        same_day_earlier_bench = log_workout(
            self.session,
            member.id,
            (self._exercise_input(bench, (6, 85)),),
            datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc),
        )
        latest_bench = log_workout(
            self.session,
            member.id,
            (self._exercise_input(bench, (5, 90)),),
            datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc),
        )
        first_bench = log_workout(
            self.session,
            member.id,
            (
                self._exercise_input(bench, (8, 80)),
                self._exercise_input(squat, (8, 100)),
            ),
            datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc),
        )
        squat_only = log_workout(
            self.session,
            member.id,
            (self._exercise_input(squat, (6, 105)),),
            datetime(2026, 9, 2, 8, 0, tzinfo=timezone.utc),
        )

        bench_history = get_exercise_history(self.session, member.id, bench.id)
        squat_history = get_exercise_history(self.session, member.id, squat.id)

        self.assertEqual(
            [workout.id for workout in bench_history],
            [latest_bench.id, same_day_earlier_bench.id, first_bench.id],
        )
        self.assertEqual(
            [workout.id for workout in squat_history],
            [squat_only.id, first_bench.id],
        )
        self.assertEqual(
            get_exercise_history(self.session, member.id, uuid4()),
            (),
        )
        self.assertEqual(
            [
                workout.id
                for workout in get_exercise_history(
                    self.session,
                    member.id,
                    bench.id,
                    limit=1,
                )
            ],
            [latest_bench.id],
        )

    def test_invalid_exercise_inputs_and_naive_time_create_no_rows(self) -> None:
        member = self._add_member("invalid-exercise@example.test")
        bench = self._add_exercise("Barbell Bench Press")
        event_time = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
        invalid_cases: tuple[
            tuple[tuple[WorkoutExerciseInput, ...], type[Exception]],
            ...,
        ] = (
            ((), ValueError),
            ((WorkoutExerciseInput(exercise_id=bench.id, sets=()),), ValueError),
            ((WorkoutExerciseInput(exercise_id=uuid4(), sets=(WorkoutSetInput(8, 80),)),), LookupError),
            ((cast(WorkoutExerciseInput, object()),), ValueError),
            (
                (
                    WorkoutExerciseInput(
                        exercise_id=cast(UUID, "not-a-uuid"),
                        sets=(WorkoutSetInput(8, 80),),
                    ),
                ),
                ValueError,
            ),
        )

        for exercises, expected_error in invalid_cases:
            with self.subTest(exercises=exercises):
                with self.assertRaises(expected_error):
                    log_workout(self.session, member.id, exercises, event_time)
                self.assertEqual(self._workout_count(member.id), 0)
                self.assertEqual(tuple(self.session.new), ())

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            log_workout(
                self.session,
                member.id,
                (self._exercise_input(bench, (8, 80)),),
                datetime(2026, 9, 1, 8, 0),
            )
        self.assertEqual(self._workout_count(member.id), 0)
        self.assertEqual(tuple(self.session.new), ())

    def test_invalid_reps_loads_and_limits_create_no_rows(self) -> None:
        member = self._add_member("invalid-set@example.test")
        bench = self._add_exercise("Barbell Bench Press")
        event_time = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)

        for reps in (0, -1, cast(int, 8.5), cast(int, True)):
            with self.subTest(reps=reps):
                exercises = (
                    WorkoutExerciseInput(
                        exercise_id=bench.id,
                        sets=(WorkoutSetInput(reps=reps, weight_kg=80),),
                    ),
                )
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    log_workout(self.session, member.id, exercises, event_time)
                self.assertEqual(self._workout_count(member.id), 0)
                self.assertEqual(tuple(self.session.new), ())

        invalid_weights = (
            -1,
            math.nan,
            math.inf,
            -math.inf,
            cast(int | float, True),
            cast(int | float, "80"),
        )
        for weight_kg in invalid_weights:
            with self.subTest(weight_kg=weight_kg):
                exercises = (
                    WorkoutExerciseInput(
                        exercise_id=bench.id,
                        sets=(WorkoutSetInput(reps=8, weight_kg=weight_kg),),
                    ),
                )
                with self.assertRaisesRegex(ValueError, "non-negative finite"):
                    log_workout(self.session, member.id, exercises, event_time)
                self.assertEqual(self._workout_count(member.id), 0)
                self.assertEqual(tuple(self.session.new), ())

        for limit in (0, -1, cast(int, True), cast(int, 1.5)):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    get_recent_workouts(self.session, member.id, limit=limit)
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    get_exercise_history(
                        self.session,
                        member.id,
                        bench.id,
                        limit=limit,
                    )


if __name__ == "__main__":
    unittest.main()
