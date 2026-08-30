"""Date-scoped reads of persisted member fitness state."""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.dates import local_date
from app.models import BodyweightLog, FoodLog, MemberProfile, WorkoutLog


@dataclass(frozen=True, slots=True)
class DayState:
    """Fitness records belonging to one member and one local calendar date."""

    member_id: UUID
    local_date: date
    bodyweight_log: BodyweightLog | None
    food_logs: tuple[FoodLog, ...]
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


def get_day_state(session: Session, member_id: UUID, day: date) -> DayState:
    """Return only records matching the requested member and local date."""

    bodyweight_log = get_bodyweight_for_day(session, member_id, day)
    food_logs = tuple(
        session.scalars(
            select(FoodLog)
            .where(FoodLog.member_id == member_id, FoodLog.log_date == day)
            .order_by(FoodLog.created_at, FoodLog.id)
        ).all()
    )
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
