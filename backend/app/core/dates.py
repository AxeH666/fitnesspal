"""Deterministic local calendar-date resolution."""

from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def local_date(instant: datetime, timezone_name: str) -> date:
    """Resolve an aware instant to its calendar date in an IANA timezone."""

    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("local_date requires a timezone-aware datetime")

    try:
        member_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        raise ValueError(f"Unknown IANA timezone: {timezone_name}") from None

    return instant.astimezone(member_timezone).date()
