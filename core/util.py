from datetime import datetime, timezone
from typing import Any, TypeVar, cast

import neotime  # type: ignore
from dateutil import parser as iso8601  # type: ignore

T = TypeVar("T")


def to_utc(dt: datetime) -> datetime:
    """
    Convert a date time (naive or timezone-aware) to a UTC-zoned datetime.
    """

    # Naive - no timezone exists so assume UTC
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(tz=timezone.utc)


def is_datetime(v: object) -> bool:
    """
    Tests if the given value is a datetime object.
    """
    return isinstance(v, (datetime, neotime.DateTime))


def normalize_datetime(dt: object) -> datetime:
    """
    Normalize a string, Neo4j Datetime, or datetime into a timezone-aware
    UTC datetime.
    """
    if isinstance(dt, str):
        d = iso8601.parse(dt)
    elif isinstance(dt, neotime.DateTime):
        d = dt.to_native()
    elif isinstance(dt, datetime):
        d = dt
    else:
        d = cast(datetime, dt)
    return to_utc(d)
