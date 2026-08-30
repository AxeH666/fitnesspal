"""Tests for deterministic recommended and active protein-target state."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal
import unittest
from typing import cast
from uuid import UUID

from sqlalchemy import Engine, Table, create_engine
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.sql.compiler import TypeCompiler

from app.db.base import Base
from app.models import BodyweightLog, Member, MemberNutritionTarget, MemberProfile
from app.services.protein_targets import (
    ProteinTargetState,
    accept_protein_recommendation,
    calculate_and_store_protein_recommendation,
    calculate_recommended_protein_target,
    get_active_protein_target,
    get_protein_target_state,
    override_protein_target,
)


@compiles(PG_UUID, "sqlite")
def _compile_postgres_uuid_for_sqlite(
    _type: object,
    _compiler: TypeCompiler,
    **_kwargs: object,
) -> str:
    """Allow the production UUID columns in the focused SQLite fixture."""

    return "CHAR(32)"


class ProteinTargetTests(unittest.TestCase):
    """Exercise the POC formula and explicit recommendation/active transitions."""

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
                cast(Table, MemberNutritionTarget.__table__),
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
        *,
        goal: str | None = "fat_loss",
    ) -> Member:
        member = Member(email=email, password_hash="test-only")
        self.session.add(member)
        self.session.flush()
        if goal is not None:
            self.session.add(
                MemberProfile(
                    member_id=member.id,
                    goal=goal,
                    activity_level="moderate",
                    training_experience="beginner",
                    timezone="Asia/Kolkata",
                )
            )
            self.session.flush()
        return member

    def _add_weight(self, member_id: UUID, day: date, weight_kg: str) -> None:
        self.session.add(
            BodyweightLog(
                member_id=member_id,
                log_date=day,
                weight_kg=Decimal(weight_kg),
                recorded_at=datetime.combine(
                    day,
                    time(hour=12),
                    tzinfo=timezone.utc,
                ),
            )
        )
        self.session.flush()

    def test_formula_is_deterministic_for_every_frozen_goal(self) -> None:
        cases = (
            ("fat_loss", 158),
            ("fat loss", 158),
            ("muscle-gain", 142),
            ("general_fitness", 126),
            ("maintenance", 126),
        )

        for goal, expected in cases:
            with self.subTest(goal=goal):
                self.assertEqual(
                    calculate_recommended_protein_target(Decimal("79"), goal),
                    expected,
                )
                self.assertEqual(
                    calculate_recommended_protein_target(Decimal("79"), goal),
                    expected,
                )

    def test_formula_rounds_to_nearest_whole_gram_half_up(self) -> None:
        cases = (
            (Decimal("79.24"), 158),
            (Decimal("79.25"), 159),
            (Decimal("79.26"), 159),
        )

        for weight_kg, expected in cases:
            with self.subTest(weight_kg=weight_kg):
                self.assertEqual(
                    calculate_recommended_protein_target(weight_kg, "fat_loss"),
                    expected,
                )

    def test_formula_rejects_invalid_weight_and_unsupported_goal(self) -> None:
        for weight_kg in (
            Decimal("0"),
            Decimal("-1"),
            Decimal("NaN"),
            Decimal("Infinity"),
            cast(Decimal, 79),
        ):
            with self.subTest(weight_kg=weight_kg):
                with self.assertRaises(ValueError):
                    calculate_recommended_protein_target(weight_kg, "fat_loss")

        for goal in ("", "strength", cast(str, None)):
            with self.subTest(goal=goal):
                with self.assertRaises(ValueError):
                    calculate_recommended_protein_target(Decimal("79"), goal)

    def test_recommendation_uses_latest_weight_and_persists_without_active(self) -> None:
        member = self._add_member("recommendation@example.test")
        self._add_weight(member.id, date(2026, 9, 2), "79.00")
        self._add_weight(member.id, date(2026, 9, 1), "95.00")

        state = calculate_and_store_protein_recommendation(self.session, member.id)

        self.assertEqual(state, ProteinTargetState(member.id, 158, None))
        self.assertIsNone(get_active_protein_target(self.session, member.id))
        self.session.commit()
        with Session(self.engine) as persisted_session:
            self.assertEqual(
                get_protein_target_state(persisted_session, member.id),
                ProteinTargetState(member.id, 158, None),
            )
            self.assertIsNone(get_active_protein_target(persisted_session, member.id))

    def test_accept_copies_stored_recommendation_without_recalculating(self) -> None:
        member = self._add_member("accept@example.test")
        self._add_weight(member.id, date(2026, 9, 1), "79.00")
        calculate_and_store_protein_recommendation(self.session, member.id)
        self._add_weight(member.id, date(2026, 9, 2), "100.00")

        state = accept_protein_recommendation(self.session, member.id)

        self.assertEqual(state, ProteinTargetState(member.id, 158, 158))
        self.assertEqual(get_active_protein_target(self.session, member.id), 158)
        self.session.commit()
        with Session(self.engine) as persisted_session:
            self.assertEqual(
                get_protein_target_state(persisted_session, member.id),
                ProteinTargetState(member.id, 158, 158),
            )

    def test_override_and_recommendation_updates_remain_separate(self) -> None:
        member = self._add_member("override@example.test")
        self._add_weight(member.id, date(2026, 9, 1), "79.00")
        calculate_and_store_protein_recommendation(self.session, member.id)

        overridden = override_protein_target(self.session, member.id, 140)
        self.assertEqual(overridden, ProteinTargetState(member.id, 158, 140))
        self.session.commit()
        with Session(self.engine) as persisted_session:
            self.assertEqual(
                get_protein_target_state(persisted_session, member.id),
                ProteinTargetState(member.id, 158, 140),
            )

        self._add_weight(member.id, date(2026, 9, 2), "80.00")
        recalculated = calculate_and_store_protein_recommendation(
            self.session,
            member.id,
        )
        self.assertEqual(recalculated, ProteinTargetState(member.id, 160, 140))

        accepted = accept_protein_recommendation(self.session, member.id)
        self.assertEqual(accepted, ProteinTargetState(member.id, 160, 160))

    def test_missing_active_and_missing_recommendation_are_safe(self) -> None:
        member = self._add_member("missing-active@example.test")
        self._add_weight(member.id, date(2026, 9, 1), "79.00")

        self.assertIsNone(get_protein_target_state(self.session, member.id))
        self.assertIsNone(get_active_protein_target(self.session, member.id))
        with self.assertRaisesRegex(LookupError, "No protein recommendation"):
            accept_protein_recommendation(self.session, member.id)
        self.assertIsNone(get_protein_target_state(self.session, member.id))

        recommendation = calculate_and_store_protein_recommendation(
            self.session,
            member.id,
        )
        self.assertEqual(recommendation.active_protein_target_g, None)
        self.assertIsNone(get_active_protein_target(self.session, member.id))

    def test_override_can_become_active_without_silent_recommendation(self) -> None:
        member = self._add_member("active-only@example.test")

        state = override_protein_target(self.session, member.id, 140)

        self.assertEqual(state, ProteinTargetState(member.id, None, 140))
        self.assertEqual(get_active_protein_target(self.session, member.id), 140)

    def test_members_remain_isolated(self) -> None:
        member_a = self._add_member("target-a@example.test", goal="fat_loss")
        member_b = self._add_member("target-b@example.test", goal="muscle_gain")
        self._add_weight(member_a.id, date(2026, 9, 1), "79.00")
        self._add_weight(member_b.id, date(2026, 9, 1), "79.00")
        calculate_and_store_protein_recommendation(self.session, member_a.id)
        calculate_and_store_protein_recommendation(self.session, member_b.id)

        accept_protein_recommendation(self.session, member_a.id)
        override_protein_target(self.session, member_b.id, 135)

        self.assertEqual(
            get_protein_target_state(self.session, member_a.id),
            ProteinTargetState(member_a.id, 158, 158),
        )
        self.assertEqual(
            get_protein_target_state(self.session, member_b.id),
            ProteinTargetState(member_b.id, 142, 135),
        )

    def test_missing_inputs_and_invalid_overrides_do_not_create_state(self) -> None:
        no_profile = self._add_member("no-profile@example.test", goal=None)
        self._add_weight(no_profile.id, date(2026, 9, 1), "79.00")
        with self.assertRaisesRegex(LookupError, "No fitness goal"):
            calculate_and_store_protein_recommendation(self.session, no_profile.id)
        self.assertIsNone(get_protein_target_state(self.session, no_profile.id))

        no_weight = self._add_member("no-weight@example.test")
        with self.assertRaisesRegex(LookupError, "No bodyweight"):
            calculate_and_store_protein_recommendation(self.session, no_weight.id)
        self.assertIsNone(get_protein_target_state(self.session, no_weight.id))

        unsupported = self._add_member("unsupported@example.test", goal="strength")
        self._add_weight(unsupported.id, date(2026, 9, 1), "79.00")
        with self.assertRaisesRegex(ValueError, "Unsupported fitness goal"):
            calculate_and_store_protein_recommendation(self.session, unsupported.id)
        self.assertIsNone(get_protein_target_state(self.session, unsupported.id))

        invalid_override = self._add_member("invalid-override@example.test")
        for target_g in (0, -1, cast(int, True), cast(int, 140.5)):
            with self.subTest(target_g=target_g):
                with self.assertRaises(ValueError):
                    override_protein_target(self.session, invalid_override.id, target_g)
                self.assertIsNone(
                    get_protein_target_state(self.session, invalid_override.id)
                )


if __name__ == "__main__":
    unittest.main()
