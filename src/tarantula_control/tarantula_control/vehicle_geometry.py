"""Vehicle-scale helpers derived from the installed Tarantula xacro model."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


_PROPERTY_TAG = "{http://www.ros.org/wiki/xacro}property"
_EXPR_RE = re.compile(r"\$\{([^}]+)\}")


@dataclass(frozen=True)
class VehicleGeometry:
    body_length: float
    body_width: float
    wheel_radius: float
    wheel_center_track: float
    overall_length: float
    overall_width: float
    total_mass: float

    @property
    def reference_length(self) -> float:
        return max(self.body_length, self.overall_length, self.overall_width)


def _description_urdf_dir() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("tarantula_description")) / "urdf"
    except Exception:
        return Path(__file__).resolve().parents[3] / "src" / "tarantula_description" / "urdf"


def _read_xacro_properties(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    values: dict[str, str] = {}
    for element in root.iter():
        if element.tag != _PROPERTY_TAG:
            continue
        name = element.attrib.get("name")
        value = element.attrib.get("value")
        if name and value is not None:
            values[name] = value
    return values


def _evaluate_property(name: str, raw: dict[str, str], resolved: dict[str, float]) -> float:
    if name in resolved:
        return resolved[name]
    if name not in raw:
        raise KeyError(name)

    def replace(match: re.Match[str]) -> str:
        ref = match.group(1).strip()
        return str(_evaluate_property(ref, raw, resolved))

    expression = _EXPR_RE.sub(replace, raw[name])
    scope = {
        "cos": math.cos,
        "sin": math.sin,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "pi": math.pi,
    }
    resolved[name] = float(eval(expression, {"__builtins__": {}}, scope))
    return resolved[name]


def load_vehicle_geometry() -> VehicleGeometry:
    urdf_dir = _description_urdf_dir()
    raw = {}
    raw.update(_read_xacro_properties(urdf_dir / "tarantula_common.xacro"))
    raw.update(_read_xacro_properties(urdf_dir / "tarantula_chassis_v2.xacro"))
    core_v3 = urdf_dir / "tarantula_core_v3.urdf.xacro"
    if core_v3.exists():
        raw.update(_read_xacro_properties(core_v3))
    resolved: dict[str, float] = {}

    body_length = _evaluate_property("body_length", raw, resolved)
    body_width = _evaluate_property("body_width", raw, resolved)
    wheel_radius = _evaluate_property("wheel_radius", raw, resolved)
    arm_length_key = "v2_arm_length" if "v2_arm_length" in raw else "arm_length"
    arm_length = _evaluate_property(arm_length_key, raw, resolved)
    arm_angle = _evaluate_property("arm_angle", raw, resolved)
    pivot_x = _evaluate_property("pivot_x", raw, resolved)
    pivot_mx = _evaluate_property("pivot_mx", raw, resolved)
    pivot_y = _evaluate_property("pivot_y", raw, resolved)
    wheel_lateral_offset = _evaluate_property("v2_wheel_lateral_offset", raw, resolved)
    body_mass = _evaluate_property("body_mass", raw, resolved)
    arm_mass = _evaluate_property("arm_mass", raw, resolved)
    wheel_mass = _evaluate_property("wheel_mass", raw, resolved)
    total_mass = body_mass + 6.0 * (arm_mass + wheel_mass)

    arm_x = arm_length * math.cos(arm_angle)
    wheel_x_positions = (
        pivot_x + arm_x,
        pivot_mx - arm_x,
        -pivot_x - arm_x,
    )
    wheel_center_track = 2.0 * (pivot_y + wheel_lateral_offset)
    return VehicleGeometry(
        body_length=body_length,
        body_width=body_width,
        wheel_radius=wheel_radius,
        wheel_center_track=wheel_center_track,
        overall_length=max(wheel_x_positions) - min(wheel_x_positions) + 2.0 * wheel_radius,
        overall_width=wheel_center_track + 2.0 * wheel_radius,
        total_mass=total_mass,
    )


VEHICLE_GEOMETRY = load_vehicle_geometry()
