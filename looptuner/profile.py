"""Parse a Nightscout profile into basal / ISF / carb-ratio schedules.

A schedule is a list of ``(start_seconds, value)`` breakpoints covering one
day; :func:`schedule_value_at` looks up the value for any second-of-day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _hhmm_to_seconds(s: str) -> int:
    parts = s.split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return h * 3600 + m * 60


def _parse_entries(raw_list) -> list[tuple[int, float]]:
    """Convert a Nightscout schedule array into sorted (seconds, value)."""
    out = []
    for item in raw_list or []:
        if "timeAsSeconds" in item and item["timeAsSeconds"] is not None:
            secs = int(float(item["timeAsSeconds"]))
        else:
            secs = _hhmm_to_seconds(item.get("time", "00:00"))
        out.append((secs, float(item["value"])))
    out.sort(key=lambda x: x[0])
    if not out:
        return out
    # Ensure the schedule starts at midnight.
    if out[0][0] != 0:
        out.insert(0, (0, out[0][1]))
    return out


@dataclass
class Profile:
    basal: list[tuple[int, float]]          # U/hr
    isf: list[tuple[int, float]]            # mg/dL per U
    carb_ratio: list[tuple[int, float]]     # g per U
    dia_hours: Optional[float] = None
    target_low: Optional[float] = None
    target_high: Optional[float] = None
    units: str = "mg/dl"
    timezone: Optional[str] = None

    @classmethod
    def from_nightscout(cls, profile_docs: list) -> "Profile":
        """Build from the list returned by ``/api/v1/profile.json``.

        Uses the most recent document's default profile store.
        """
        if not profile_docs:
            raise ValueError("no profile documents returned by Nightscout")
        doc = profile_docs[0]
        default_name = doc.get("defaultProfile")
        store = doc.get("store", {})
        if default_name and default_name in store:
            p = store[default_name]
        elif store:
            p = next(iter(store.values()))
        else:
            p = doc  # very old single-profile format

        isf = _parse_entries(p.get("sens"))
        return cls(
            basal=_parse_entries(p.get("basal")),
            isf=isf,
            carb_ratio=_parse_entries(p.get("carbratio")),
            dia_hours=_to_float(p.get("dia")),
            target_low=_first_value(p.get("target_low")),
            target_high=_first_value(p.get("target_high")),
            units=(p.get("units") or "mg/dl").lower(),
            timezone=p.get("timezone"),
        )

    @classmethod
    def from_schedule_list(cls, basal_schedule: list) -> "Profile":
        """Build a minimal profile from a fallback basal schedule.

        ``basal_schedule`` is a list of ``{"start": "HH:MM", "rate": U/hr}``.
        ISF / carb-ratio are left empty (they are estimated, not required as
        priors at the schedule level).
        """
        basal = sorted(
            (_hhmm_to_seconds(b["start"]), float(b["rate"])) for b in basal_schedule
        )
        if basal and basal[0][0] != 0:
            basal.insert(0, (0, basal[0][1]))
        return cls(basal=basal, isf=[], carb_ratio=[])


def schedule_value_at(schedule: list[tuple[int, float]], second_of_day: float) -> float:
    """Value in effect at ``second_of_day`` (step function)."""
    if not schedule:
        raise ValueError("empty schedule")
    value = schedule[0][1]
    for secs, val in schedule:
        if second_of_day >= secs:
            value = val
        else:
            break
    return value


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _first_value(entries):
    parsed = _parse_entries(entries) if entries else []
    return parsed[0][1] if parsed else None
