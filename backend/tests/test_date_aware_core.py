"""Tests for deterministic local dates and date-scoped member state."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal
import unittest
from typing import cast
from uuid import UUID

from sqlalchemy import Engine, Table, create_engine, func, select
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
from app.services.member_state import (
    DayState,
    get_bodyweight_for_day,
    get_day_state,
    get_latest_bodyweight,
    get_today_state,
    log_bodyweight,
)


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
            recorded_at=datetime.combine(
                day,
                time(hour=12),
                tzinfo=timezone.utc,
            ),
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

    def _bodyweight_count(self, member_id: UUID) -> int:
        count = self.session.scalar(
            select(func.count())
            .select_from(BodyweightLog)
            .where(BodyweightLog.member_id == member_id)
        )
        assert count is not None
        return count

    def assert_recorded_at_matches(
        self,
        bodyweight_log: BodyweightLog,
        expected: datetime,
    ) -> None:
        actual = bodyweight_log.recorded_at
        if actual.tzinfo is None:
            actual = actual.replace(tzinfo=timezone.utc)
        self.assertEqual(actual, expected.astimezone(timezone.utc))

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

    def test_first_weight_log_persists_with_event_time_and_local_date(self) -> None:
        member = self._add_member("first-weight@example.test")
        member_id = member.id
        event_time = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)

        logged = log_bodyweight(
            self.session,
            member_id,
            Decimal("79.40"),
            event_time,
        )
        logged_id = logged.id
        self.session.expunge_all()
        persisted = get_bodyweight_for_day(
            self.session,
            member_id,
            date(2026, 8, 31),
        )

        self.assertIsNotNone(persisted)
        assert persisted is not None
        self.assertEqual(persisted.id, logged_id)
        self.assertEqual(persisted.member_id, member_id)
        self.assertEqual(persisted.weight_kg, Decimal("79.40"))
        self.assertEqual(persisted.log_date, date(2026, 8, 31))
        self.assert_recorded_at_matches(persisted, event_time)

    def test_second_weight_on_same_local_day_updates_existing_row(self) -> None:
        member = self._add_member("same-day@example.test")
        first_time = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)
        second_time = datetime(2026, 8, 31, 17, 0, tzinfo=timezone.utc)

        first = log_bodyweight(
            self.session,
            member.id,
            Decimal("79.40"),
            first_time,
        )
        first_id = first.id
        second = log_bodyweight(
            self.session,
            member.id,
            Decimal("79.10"),
            second_time,
        )

        self.assertEqual(second.id, first_id)
        self.assertEqual(second.log_date, date(2026, 8, 31))
        self.assertEqual(second.weight_kg, Decimal("79.10"))
        self.assert_recorded_at_matches(second, second_time)
        self.assertEqual(self._bodyweight_count(member.id), 1)

    def test_local_midnight_creates_next_day_without_changing_day_a(self) -> None:
        member = self._add_member("two-days@example.test")
        before_midnight = datetime(2026, 8, 30, 18, 29, 59, tzinfo=timezone.utc)
        at_midnight = datetime(2026, 8, 30, 18, 30, tzinfo=timezone.utc)

        day_a_log = log_bodyweight(
            self.session,
            member.id,
            Decimal("80.00"),
            before_midnight,
        )
        day_a_id = day_a_log.id
        day_b_log = log_bodyweight(
            self.session,
            member.id,
            Decimal("79.80"),
            at_midnight,
        )
        persisted_day_a = get_bodyweight_for_day(
            self.session,
            member.id,
            date(2026, 8, 30),
        )
        persisted_day_b = get_bodyweight_for_day(
            self.session,
            member.id,
            date(2026, 8, 31),
        )

        self.assertIsNotNone(persisted_day_a)
        self.assertIsNotNone(persisted_day_b)
        assert persisted_day_a is not None
        assert persisted_day_b is not None
        self.assertEqual(persisted_day_a.id, day_a_id)
        self.assertEqual(persisted_day_a.weight_kg, Decimal("80.00"))
        self.assert_recorded_at_matches(persisted_day_a, before_midnight)
        self.assertEqual(persisted_day_b.id, day_b_log.id)
        self.assertEqual(persisted_day_b.weight_kg, Decimal("79.80"))
        self.assertNotEqual(persisted_day_a.id, persisted_day_b.id)
        self.assertEqual(self._bodyweight_count(member.id), 2)

    def test_latest_bodyweight_uses_newest_day_not_write_order(self) -> None:
        member = self._add_member("latest@example.test")
        log_bodyweight(
            self.session,
            member.id,
            Decimal("79.80"),
            datetime(2026, 8, 29, 20, 0, tzinfo=timezone.utc),
        )
        newer = log_bodyweight(
            self.session,
            member.id,
            Decimal("78.90"),
            datetime(2026, 8, 31, 20, 0, tzinfo=timezone.utc),
        )
        newer_id = newer.id
        log_bodyweight(
            self.session,
            member.id,
            Decimal("79.40"),
            datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc),
        )

        latest = get_latest_bodyweight(self.session, member.id)

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.id, newer_id)
        self.assertEqual(latest.log_date, date(2026, 9, 1))
        self.assertEqual(latest.weight_kg, Decimal("78.90"))

    def test_bodyweight_reads_and_updates_are_member_scoped(self) -> None:
        member_a = self._add_member("weight-a@example.test")
        member_b = self._add_member("weight-b@example.test")
        same_day = datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc)
        member_a_log = log_bodyweight(
            self.session,
            member_a.id,
            Decimal("79.00"),
            same_day,
        )
        member_a_id = member_a_log.id
        member_b_day_a = log_bodyweight(
            self.session,
            member_b.id,
            Decimal("91.00"),
            same_day,
        )
        member_b_day_a_id = member_b_day_a.id
        log_bodyweight(
            self.session,
            member_a.id,
            Decimal("78.80"),
            datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc),
        )
        member_b_day_b = log_bodyweight(
            self.session,
            member_b.id,
            Decimal("90.80"),
            datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc),
        )

        member_a_day = get_bodyweight_for_day(
            self.session,
            member_a.id,
            date(2026, 8, 30),
        )
        member_b_day = get_bodyweight_for_day(
            self.session,
            member_b.id,
            date(2026, 8, 30),
        )
        latest_a = get_latest_bodyweight(self.session, member_a.id)
        latest_b = get_latest_bodyweight(self.session, member_b.id)

        self.assertIsNotNone(member_a_day)
        self.assertIsNotNone(member_b_day)
        self.assertIsNotNone(latest_a)
        self.assertIsNotNone(latest_b)
        assert member_a_day is not None
        assert member_b_day is not None
        assert latest_a is not None
        assert latest_b is not None
        self.assertEqual(member_a_day.id, member_a_id)
        self.assertEqual(member_a_day.weight_kg, Decimal("78.80"))
        self.assertEqual(member_b_day.id, member_b_day_a_id)
        self.assertEqual(member_b_day.weight_kg, Decimal("91.00"))
        self.assertEqual(latest_a.id, member_a_id)
        self.assertEqual(latest_b.id, member_b_day_b.id)

    def test_same_instant_uses_each_members_timezone_when_logging(self) -> None:
        kolkata_member = self._add_member("log-kolkata@example.test", "Asia/Kolkata")
        new_york_member = self._add_member(
            "log-new-york@example.test",
            "America/New_York",
        )
        event_time = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)

        kolkata_log = log_bodyweight(
            self.session,
            kolkata_member.id,
            Decimal("79.00"),
            event_time,
        )
        new_york_log = log_bodyweight(
            self.session,
            new_york_member.id,
            Decimal("91.00"),
            event_time,
        )

        self.assertEqual(kolkata_log.log_date, date(2026, 8, 31))
        self.assertEqual(new_york_log.log_date, date(2026, 8, 30))
        self.assertIsNone(
            get_bodyweight_for_day(
                self.session,
                kolkata_member.id,
                date(2026, 8, 30),
            )
        )
        self.assertIsNone(
            get_bodyweight_for_day(
                self.session,
                new_york_member.id,
                date(2026, 8, 31),
            )
        )

    def test_zero_weight_is_rejected_without_side_effects(self) -> None:
        member = self._add_member("zero@example.test")

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            log_bodyweight(
                self.session,
                member.id,
                Decimal("0"),
                datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc),
            )

        self.assertEqual(len(self.session.new), 0)
        self.assertEqual(self._bodyweight_count(member.id), 0)

    def test_negative_weight_is_rejected_without_side_effects(self) -> None:
        member = self._add_member("negative@example.test")

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            log_bodyweight(
                self.session,
                member.id,
                Decimal("-1"),
                datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc),
            )

        self.assertEqual(len(self.session.new), 0)
        self.assertEqual(self._bodyweight_count(member.id), 0)

    def test_non_finite_weights_are_rejected_without_side_effects(self) -> None:
        member = self._add_member("non-finite@example.test")
        event_time = datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc)

        for weight_kg in (
            Decimal("NaN"),
            Decimal("Infinity"),
            Decimal("-Infinity"),
        ):
            with self.subTest(weight_kg=weight_kg):
                with self.assertRaisesRegex(ValueError, "finite value"):
                    log_bodyweight(
                        self.session,
                        member.id,
                        weight_kg,
                        event_time,
                    )

                self.assertEqual(len(self.session.new), 0)
                self.assertEqual(self._bodyweight_count(member.id), 0)

    def test_naive_event_time_is_rejected_without_side_effects(self) -> None:
        member = self._add_member("naive-time@example.test")

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            log_bodyweight(
                self.session,
                member.id,
                Decimal("79.00"),
                datetime(2026, 8, 30, 10, 0),
            )

        self.assertEqual(len(self.session.new), 0)
        self.assertEqual(self._bodyweight_count(member.id), 0)

    def test_logged_weights_flow_into_today_and_requested_day_state(self) -> None:
        member = self._add_member("weight-state@example.test")
        day_a_log = log_bodyweight(
            self.session,
            member.id,
            Decimal("80.00"),
            datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc),
        )
        day_b_time = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)
        day_b_log = log_bodyweight(
            self.session,
            member.id,
            Decimal("79.80"),
            day_b_time,
        )

        today_state = get_today_state(
            self.session,
            member.id,
            now=day_b_time,
        )
        day_a_state = get_day_state(
            self.session,
            member.id,
            date(2026, 8, 30),
        )

        self.assertIsNotNone(today_state.bodyweight_log)
        self.assertIsNotNone(day_a_state.bodyweight_log)
        assert today_state.bodyweight_log is not None
        assert day_a_state.bodyweight_log is not None
        self.assertEqual(today_state.bodyweight_log.id, day_b_log.id)
        self.assertEqual(today_state.bodyweight_log.weight_kg, Decimal("79.80"))
        self.assertEqual(day_a_state.bodyweight_log.id, day_a_log.id)
        self.assertEqual(day_a_state.bodyweight_log.weight_kg, Decimal("80.00"))


if __name__ == "__main__":
    unittest.main()
