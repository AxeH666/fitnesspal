"""Tests for deterministic local dates and date-scoped member state."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import unittest
from typing import cast
from uuid import UUID

from sqlalchemy import Engine, Table, create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.sql.compiler import TypeCompiler

from app.core.dates import local_date
from app.db.base import Base
from app.models import (
    BodyweightLog,
    FoodLog,
    Member,
    MemberProfile,
    WorkoutLog,
    WorkoutPlan,
)
from app.services.member_state import DayState, get_day_state, get_today_state


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(
    _type: JSONB,
    _compiler: TypeCompiler,
    **_kwargs: object,
) -> str:
    """Allow the production JSONB model columns in this focused SQLite fixture."""

    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_postgres_uuid_for_sqlite(
    _type: object,
    _compiler: TypeCompiler,
    **_kwargs: object,
) -> str:
    """Allow the production UUID model columns in this focused SQLite fixture."""

    return "CHAR(32)"


class DateAwareCoreTests(unittest.TestCase):
    """Exercise the real state queries without requiring a shared database."""

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
                cast(Table, WorkoutPlan.__table__),
                cast(Table, WorkoutLog.__table__),
            ],
        )
        self.session = Session(self.engine)

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

    def _add_day_records(
        self,
        member_id: UUID,
        day: date,
        marker: str,
        weight_kg: str,
    ) -> tuple[BodyweightLog, FoodLog, WorkoutLog]:
        weight = BodyweightLog(
            member_id=member_id,
            log_date=day,
            weight_kg=Decimal(weight_kg),
            note=marker,
        )
        food = FoodLog(
            member_id=member_id,
            log_date=day,
            items=[{"marker": marker}],
            totals={"calories": 1},
            note=marker,
        )
        workout = WorkoutLog(
            member_id=member_id,
            log_date=day,
            exercises=[{"marker": marker}],
            note=marker,
        )
        self.session.add_all([weight, food, workout])
        self.session.flush()
        return weight, food, workout

    def assert_state_contains_only(
        self,
        state: DayState,
        expected: tuple[BodyweightLog, FoodLog, WorkoutLog],
    ) -> None:
        weight, food, workout = expected
        self.assertIsNotNone(state.bodyweight_log)
        assert state.bodyweight_log is not None
        self.assertEqual(state.bodyweight_log.id, weight.id)
        self.assertEqual([log.id for log in state.food_logs], [food.id])
        self.assertEqual([log.id for log in state.workout_logs], [workout.id])
        self.assertEqual(state.bodyweight_log.log_date, state.local_date)
        self.assertTrue(all(log.log_date == state.local_date for log in state.food_logs))
        self.assertTrue(all(log.log_date == state.local_date for log in state.workout_logs))

    def test_local_date_uses_the_requested_timezone(self) -> None:
        instant = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)

        self.assertEqual(local_date(instant, "Asia/Kolkata"), date(2026, 8, 31))
        self.assertEqual(local_date(instant, "America/New_York"), date(2026, 8, 30))

    def test_local_date_rejects_naive_datetime(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            local_date(datetime(2026, 8, 30, 20, 0), "Asia/Kolkata")

    def test_local_date_rejects_unknown_timezone(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown IANA timezone"):
            local_date(
                datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc),
                "Not/A_Timezone",
            )

    def test_day_a_remains_separate_after_day_b_is_added(self) -> None:
        member = self._add_member("member@example.test")
        other_member = self._add_member("other@example.test")
        day_a = date(2026, 8, 30)
        day_b = date(2026, 8, 31)
        expected_a = self._add_day_records(member.id, day_a, "day-a", "80.00")
        expected_other = self._add_day_records(
            other_member.id,
            day_a,
            "other-member",
            "90.00",
        )

        state_before_day_b = get_day_state(self.session, member.id, day_a)
        expected_b = self._add_day_records(member.id, day_b, "day-b", "81.00")
        state_a = get_day_state(self.session, member.id, day_a)
        state_b = get_day_state(self.session, member.id, day_b)
        state_other = get_day_state(self.session, other_member.id, day_a)

        self.assert_state_contains_only(state_before_day_b, expected_a)
        self.assert_state_contains_only(state_a, expected_a)
        self.assert_state_contains_only(state_b, expected_b)
        self.assert_state_contains_only(state_other, expected_other)

    def test_today_switches_state_at_member_local_midnight(self) -> None:
        member = self._add_member("midnight@example.test")
        day_a = date(2026, 8, 30)
        day_b = date(2026, 8, 31)
        expected_a = self._add_day_records(member.id, day_a, "day-a", "80.00")
        expected_b = self._add_day_records(member.id, day_b, "day-b", "81.00")

        before_midnight = get_today_state(
            self.session,
            member.id,
            now=datetime(2026, 8, 30, 18, 29, 59, tzinfo=timezone.utc),
        )
        at_midnight = get_today_state(
            self.session,
            member.id,
            now=datetime(2026, 8, 30, 18, 30, tzinfo=timezone.utc),
        )

        self.assert_state_contains_only(before_midnight, expected_a)
        self.assert_state_contains_only(at_midnight, expected_b)

    def test_today_uses_the_requested_members_timezone(self) -> None:
        kolkata_member = self._add_member("kolkata@example.test", "Asia/Kolkata")
        new_york_member = self._add_member(
            "new-york@example.test",
            "America/New_York",
        )
        fixed_instant = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)
        kolkata_records = self._add_day_records(
            kolkata_member.id,
            date(2026, 8, 31),
            "kolkata",
            "80.00",
        )
        new_york_records = self._add_day_records(
            new_york_member.id,
            date(2026, 8, 30),
            "new-york",
            "90.00",
        )

        kolkata_today = get_today_state(
            self.session,
            kolkata_member.id,
            now=fixed_instant,
        )
        new_york_today = get_today_state(
            self.session,
            new_york_member.id,
            now=fixed_instant,
        )

        self.assert_state_contains_only(kolkata_today, kolkata_records)
        self.assert_state_contains_only(new_york_today, new_york_records)

    def test_empty_day_returns_empty_collections(self) -> None:
        member = self._add_member("empty@example.test")
        requested_day = date(2026, 9, 1)

        state = get_day_state(self.session, member.id, requested_day)

        self.assertEqual(state.local_date, requested_day)
        self.assertIsNone(state.bodyweight_log)
        self.assertEqual(state.food_logs, ())
        self.assertEqual(state.workout_logs, ())


if __name__ == "__main__":
    unittest.main()
