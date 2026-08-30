"""Date-scoped reads of persisted member fitness state."""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from uuid import UUID

from sqlalchemy import select
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


def get_day_state(session: Session, member_id: UUID, day: date) -> DayState:
    """Return only records matching the requested member and local date."""

    bodyweight_log = session.scalar(
        select(BodyweightLog).where(
            BodyweightLog.member_id == member_id,
            BodyweightLog.log_date == day,
        )
    )
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

    timezone_name = session.scalar(
        select(MemberProfile.timezone).where(MemberProfile.member_id == member_id)
    )
    if timezone_name is None:
        raise LookupError(f"No timezone found for member {member_id}")

    instant = now if now is not None else datetime.now(timezone.utc)
    return get_day_state(session, member_id, local_date(instant, timezone_name))
