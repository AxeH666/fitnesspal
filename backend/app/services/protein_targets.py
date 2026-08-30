"""Deterministic recommended and active protein-target state."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models import MemberNutritionTarget, MemberProfile
from app.services.member_state import get_latest_bodyweight


_PROTEIN_GRAMS_PER_KG = {
    "fat_loss": Decimal("2.0"),
    "muscle_gain": Decimal("1.8"),
    "general_fitness": Decimal("1.6"),
    "maintenance": Decimal("1.6"),
}


@dataclass(frozen=True, slots=True)
class ProteinTargetState:
    """Stored recommendation and the separately selected active target."""

    member_id: UUID
    recommended_protein_target_g: int | None
    active_protein_target_g: int | None


def _normalized_goal(goal: str) -> str:
    """Return the canonical POC goal key without inventing new categories."""

    normalized = "_".join(goal.strip().lower().replace("-", " ").split())
    if normalized not in _PROTEIN_GRAMS_PER_KG:
        raise ValueError(f"Unsupported fitness goal for protein recommendation: {goal}")
    return normalized


def calculate_recommended_protein_target(weight_kg: Decimal, goal: str) -> int:
    """Calculate the frozen POC recommendation from weight and goal only."""

    if not isinstance(weight_kg, Decimal) or not weight_kg.is_finite() or weight_kg <= 0:
        raise ValueError("weight_kg must be a finite Decimal greater than zero")
    if not isinstance(goal, str):
        raise ValueError("goal must be a supported fitness-goal string")

    recommendation = weight_kg * _PROTEIN_GRAMS_PER_KG[_normalized_goal(goal)]
    return int(recommendation.to_integral_value(rounding=ROUND_HALF_UP))


def get_protein_target_state(
    session: Session,
    member_id: UUID,
) -> ProteinTargetState | None:
    """Return stored protein state without loading unrelated legacy columns."""

    row = session.execute(
        select(
            MemberNutritionTarget.member_id,
            MemberNutritionTarget.recommended_protein_target_g,
            MemberNutritionTarget.active_protein_target_g,
        ).where(MemberNutritionTarget.member_id == member_id)
    ).one_or_none()
    if row is None:
        return None

    return ProteinTargetState(
        member_id=row[0],
        recommended_protein_target_g=row[1],
        active_protein_target_g=row[2],
    )


def get_active_protein_target(session: Session, member_id: UUID) -> int | None:
    """Return only the explicitly selected active target, never the recommendation."""

    state = get_protein_target_state(session, member_id)
    return None if state is None else state.active_protein_target_g


def _stored_state(session: Session, member_id: UUID) -> ProteinTargetState:
    """Return state after a successful mutation."""

    state = get_protein_target_state(session, member_id)
    if state is None:
        raise RuntimeError(f"Protein target state was not stored for member {member_id}")
    return state


def calculate_and_store_protein_recommendation(
    session: Session,
    member_id: UUID,
) -> ProteinTargetState:
    """Calculate from persisted member data and store recommendation only."""

    goal = session.scalar(
        select(MemberProfile.goal).where(MemberProfile.member_id == member_id)
    )
    if goal is None:
        raise LookupError(f"No fitness goal found for member {member_id}")

    bodyweight = get_latest_bodyweight(session, member_id)
    if bodyweight is None:
        raise LookupError(f"No bodyweight found for member {member_id}")

    recommendation = calculate_recommended_protein_target(bodyweight.weight_kg, goal)
    calculated_at = datetime.now(timezone.utc)
    table = MemberNutritionTarget.__table__
    insert_statement = insert(MemberNutritionTarget).values(
        member_id=member_id,
        recommended_protein_target_g=recommendation,
        calculated_at=calculated_at,
    )
    session.execute(
        insert_statement.on_conflict_do_update(
            index_elements=[table.c.member_id],
            set_={
                table.c.recommended_protein_target_g: (
                    insert_statement.excluded.recommended_protein_target_g
                ),
                table.c.calculated_at: insert_statement.excluded.calculated_at,
                table.c.updated_at: func.now(),
            },
        )
    )
    return _stored_state(session, member_id)


def accept_protein_recommendation(
    session: Session,
    member_id: UUID,
) -> ProteinTargetState:
    """Make the currently stored recommendation active by explicit action."""

    table = MemberNutritionTarget.__table__
    updated_member_id = session.scalar(
        update(MemberNutritionTarget)
        .where(
            table.c.member_id == member_id,
            table.c.recommended_protein_target_g.is_not(None),
        )
        .values(
            {
                table.c.protein_target_g: table.c.recommended_protein_target_g,
                table.c.updated_at: func.now(),
            }
        )
        .returning(table.c.member_id)
    )
    if updated_member_id is None:
        raise LookupError(f"No protein recommendation found for member {member_id}")
    return _stored_state(session, member_id)


def override_protein_target(
    session: Session,
    member_id: UUID,
    target_g: int,
) -> ProteinTargetState:
    """Make a positive whole-gram user choice active without changing recommendation."""

    if isinstance(target_g, bool) or not isinstance(target_g, int) or target_g <= 0:
        raise ValueError("target_g must be a positive integer")

    table = MemberNutritionTarget.__table__
    insert_statement = insert(MemberNutritionTarget).values(
        member_id=member_id,
        protein_target_g=target_g,
    )
    session.execute(
        insert_statement.on_conflict_do_update(
            index_elements=[table.c.member_id],
            set_={
                table.c.protein_target_g: insert_statement.excluded.protein_target_g,
                table.c.updated_at: func.now(),
            },
        )
    )
    return _stored_state(session, member_id)
