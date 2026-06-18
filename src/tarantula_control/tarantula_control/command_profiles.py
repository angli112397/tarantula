"""Reusable command profiles for local motion and Nav-style evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .vehicle_geometry import VEHICLE_GEOMETRY


REFERENCE_LENGTH = VEHICLE_GEOMETRY.reference_length
DEMO_DRIVE_VX = 0.18
DEMO_TURN_WZ = 0.12


@dataclass(frozen=True)
class CommandSegment:
    name: str
    vx: float
    wz: float
    duration_s: float


_PRIMITIVE_PROFILE = (
    ("stop", 0.0, 0.0, None),
    ("turn_left_from_drive_cmd", 0.1, 0.15, None),
    ("drive_after_left", 0.1, 0.0, None),
    ("turn_right_from_drive_cmd", 0.1, -0.15, None),
    ("drive_after_right", 0.1, 0.0, None),
    ("backward", -0.1, 0.0, None),
    ("turn_left_authority", 0.0, 0.25, None),
    ("turn_right_authority", 0.0, -0.25, None),
    ("final_stop", 0.0, 0.0, None),
)

_NAVI_POINTS = (
    (0.0, 0.0),
    (3.0, 0.0),
    (4.5, 1.8),
    (6.8, 1.8),
    (8.0, 0.5),
    (10.0, 0.5),
)

PROFILE_CHOICES = ("navi", "primitive", "both")


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _polyline_profile(
    prefix: str,
    points: tuple[tuple[float, float], ...],
    *,
    drive_vx: float,
    turn_wz: float,
) -> list[CommandSegment]:
    if len(points) < 2:
        return []

    segments = [CommandSegment(f"{prefix}_initial_stop", 0.0, 0.0, 0.5)]
    heading = 0.0
    side_idx = 1
    turn_idx = 1
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance <= 1.0e-6:
            continue
        target_heading = math.atan2(dy, dx)
        heading_delta = _wrap_angle(target_heading - heading)
        if abs(heading_delta) > 1.0e-4:
            signed_wz = math.copysign(turn_wz, heading_delta)
            segments.append(
                CommandSegment(
                    f"{prefix}_turn_{turn_idx}",
                    0.0,
                    signed_wz,
                    abs(heading_delta) / turn_wz,
                )
            )
            turn_idx += 1
        segments.append(
            CommandSegment(
                f"{prefix}_side_{side_idx}",
                drive_vx,
                0.0,
                distance / drive_vx,
            )
        )
        side_idx += 1
        heading = target_heading
    final_heading_delta = _wrap_angle(-heading)
    if abs(final_heading_delta) > 1.0e-4:
        signed_wz = math.copysign(turn_wz, final_heading_delta)
        segments.append(
            CommandSegment(
                f"{prefix}_turn_{turn_idx}",
                0.0,
                signed_wz,
                abs(final_heading_delta) / turn_wz,
            )
        )
    segments.append(CommandSegment(f"{prefix}_final_stop", 0.0, 0.0, 0.5))
    return segments


def _scaled_points(points: tuple[tuple[float, float], ...], scale: float) -> tuple[tuple[float, float], ...]:
    return tuple((x * scale, y * scale) for x, y in points)


def profile_sequence(profile: str, default_duration_s: float = 4.0) -> list[CommandSegment]:
    if profile == "navi":
        return _polyline_profile(
            "navi",
            _scaled_points(_NAVI_POINTS, REFERENCE_LENGTH),
            drive_vx=DEMO_DRIVE_VX,
            turn_wz=DEMO_TURN_WZ,
        )
    elif profile == "primitive":
        source = _PRIMITIVE_PROFILE
    elif profile == "both":
        return profile_sequence("navi", default_duration_s) + profile_sequence("primitive", default_duration_s)
    else:
        raise ValueError(f"unknown profile {profile!r}")

    return [
        CommandSegment(
            name=name,
            vx=float(vx),
            wz=float(wz),
            duration_s=float(duration_s if duration_s is not None else default_duration_s),
        )
        for name, vx, wz, duration_s in source
    ]


def parse_route_specs(
    specs: list[str],
    *,
    profile: str = "navi",
    default_duration_s: float = 4.0,
) -> list[CommandSegment]:
    if not specs:
        return profile_sequence(profile, default_duration_s)

    sequence = []
    for spec in specs:
        parts = spec.split(",")
        if len(parts) not in (3, 4):
            raise ValueError(f"bad --segment {spec!r}; expected name,vx,wz[,duration_s]")
        duration_s = float(parts[3]) if len(parts) == 4 else float(default_duration_s)
        sequence.append(CommandSegment(parts[0], float(parts[1]), float(parts[2]), duration_s))
    return sequence
