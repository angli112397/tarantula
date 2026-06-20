import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Bump whenever a change here would make an already-generated terrain dir
# behave differently if re-run today (e.g. SurfaceProps/DEFAULT_SURFACE
# tuning, export_world_sdf's surround_copies, generator.py's height-field
# logic). Embedded in metadata.json by generator.py/nav_maze.py so
# scripts/check_terrain_freshness.py can flag already-generated dirs that
# predate the change instead of silently running stale physics/geometry
# (this bit us twice now: once for gazebo_demo/42's contact surface, once
# for rl_curriculum's unsmoothed tile-edge cliffs -- see _generate_rl_curriculum).
GENERATOR_SCHEMA_VERSION = 2


def _write_png(path: Path, height: np.ndarray) -> None:
    lo = float(height.min())
    hi = float(height.max())
    span = max(hi - lo, 1e-6)
    img = ((height - lo) / span * 255.0).clip(0, 255).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(img, mode="L").save(path)
    except Exception:
        pgm_path = path.with_suffix(".pgm")
        with pgm_path.open("wb") as f:
            f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode("ascii"))
            f.write(img.tobytes())


def export_height_assets(out_dir: Path, height: np.ndarray, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "height.npy", height.astype(np.float32))
    _write_png(out_dir / "height.png", height)
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def flip_for_map_server(array: np.ndarray) -> np.ndarray:
    # ROS map_server treats image row 0 as the map's top row (max y / north);
    # our arrays are stored south-to-north. Every PGM writer below goes
    # through this one function so the row convention only lives in one place.
    return array[::-1, :]


def write_pgm(path: Path, image: np.ndarray) -> None:
    with path.open("wb") as f:
        f.write(f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
        f.write(image.tobytes())


def occupancy_to_pgm_image(occ: np.ndarray) -> np.ndarray:
    return np.where(flip_for_map_server(occ), 0, 254).astype(np.uint8)


def scaled_cost_to_pgm_image(cost: np.ndarray) -> np.ndarray:
    # Cost convention is 0/free .. 100/lethal; map_server's "scale" mode reads
    # white as free and black as occupied.
    flipped = flip_for_map_server(cost).astype(np.float32)
    return np.rint(254.0 - np.clip(flipped, 0.0, 100.0) / 100.0 * 254.0).astype(np.uint8)


def write_map_yaml(
    path: Path,
    image_name: str,
    mode: str,
    metadata: dict,
    *,
    occupied_thresh: float,
    free_thresh: float,
) -> None:
    path.write_text(
        "\n".join(
            [
                f"image: {image_name}",
                f"mode: {mode}",
                f"resolution: {float(metadata['resolution']):.6f}",
                f"origin: [{-float(metadata['size_x']) / 2.0:.6f}, {-float(metadata['size_y']) / 2.0:.6f}, 0.0]",
                "negate: 0",
                f"occupied_thresh: {occupied_thresh}",
                f"free_thresh: {free_thresh}",
                "",
            ]
        ),
        encoding="ascii",
    )


def export_obj(out_dir: Path, height: np.ndarray, resolution: float) -> Path:
    obj_path = out_dir / "terrain.obj"
    mtl_path = out_dir / "terrain.mtl"
    ny, nx = height.shape
    x0 = -(nx - 1) * resolution / 2.0
    y0 = -(ny - 1) * resolution / 2.0
    gy, gx = np.gradient(height, resolution, resolution)
    normals = np.dstack((-gx, -gy, np.ones_like(height)))
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(norms, 1e-9)

    with mtl_path.open("w", encoding="ascii") as f:
        f.write("newmtl terrain_mat\n")
        f.write("Ka 0.34 0.35 0.33\n")
        f.write("Kd 0.42 0.43 0.40\n")
        f.write("Ks 0.05 0.05 0.05\n")

    with obj_path.open("w", encoding="ascii") as f:
        f.write("mtllib terrain.mtl\n")
        f.write("usemtl terrain_mat\n")
        for iy in range(ny):
            y = y0 + iy * resolution
            for ix in range(nx):
                x = x0 + ix * resolution
                f.write(f"v {x:.6f} {y:.6f} {height[iy, ix]:.6f}\n")
        for iy in range(ny):
            for ix in range(nx):
                nx_, ny_, nz_ = normals[iy, ix]
                f.write(f"vn {nx_:.6f} {ny_:.6f} {nz_:.6f}\n")
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                v0 = iy * nx + ix + 1
                v1 = v0 + 1
                v2 = v0 + nx
                v3 = v2 + 1
                f.write(f"f {v0}//{v0} {v1}//{v1} {v3}//{v3}\n")
                f.write(f"f {v0}//{v0} {v3}//{v3} {v2}//{v2}\n")
    return obj_path


@dataclass(frozen=True)
class SurfaceProps:
    """ODE friction/contact tuning shared by every collision geometry we export.

    Without slip/contact damping, Gazebo's ODE defaults leave thin boxes and
    meshes essentially frictionless (skid-steer wheels can't generate a yaw
    moment), so this is threaded through every exporter instead of being a
    one-off block on a single geometry type.
    """

    mu: float = 1.50
    mu2: float = 1.50
    slip1: float = 0.001
    slip2: float = 0.001
    kp: float = 1_000_000.0
    kd: float = 10.0
    max_vel: float = 0.2
    min_depth: float = 0.001


DEFAULT_SURFACE = SurfaceProps()


def _surface_sdf(surface: SurfaceProps) -> str:
    return f"""<surface>
          <friction><ode><mu>{surface.mu}</mu><mu2>{surface.mu2}</mu2><slip1>{surface.slip1}</slip1><slip2>{surface.slip2}</slip2></ode></friction>
          <contact><ode><kp>{surface.kp}</kp><kd>{surface.kd}</kd><max_vel>{surface.max_vel}</max_vel><min_depth>{surface.min_depth}</min_depth></ode></contact>
        </surface>"""


def export_terrain_sdf(out_dir: Path, obj_path: Path, surface: SurfaceProps = DEFAULT_SURFACE) -> Path:
    sdf_path = out_dir / "terrain.sdf"
    mesh_uri = obj_path.resolve().as_uri()
    sdf_path.write_text(
        f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="generated_heightmap_terrain">
    <static>true</static>
    <link name="terrain_link">
      <collision name="terrain_collision">
        <geometry>
          <mesh><uri>{mesh_uri}</uri></mesh>
        </geometry>
        {_surface_sdf(surface)}
      </collision>
      <visual name="terrain_visual">
        <geometry>
          <mesh><uri>{mesh_uri}</uri></mesh>
        </geometry>
        <material>
          <ambient>0.34 0.35 0.33 1</ambient>
          <diffuse>0.42 0.43 0.40 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""",
        encoding="utf-8",
    )
    return sdf_path


def _terrain_visual_model(obj_path: Path) -> str:
    mesh_uri = obj_path.resolve().as_uri()
    return f"""
    <model name="terrain_visual_only">
      <static>true</static>
      <link name="terrain_visual_link">
        <visual name="terrain_visual">
          <geometry>
            <mesh><uri>{mesh_uri}</uri></mesh>
          </geometry>
          <material>
            <ambient>0.30 0.36 0.32 0.78</ambient>
            <diffuse>0.36 0.44 0.38 0.78</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def _box_model(name: str, pose: str, size: str, color: str, surface: SurfaceProps = DEFAULT_SURFACE) -> str:
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="link">
        <collision name="c">
          <geometry><box><size>{size}</size></box></geometry>
          {_surface_sdf(surface)}
        </collision>
        <visual name="v"><geometry><box><size>{size}</size></box></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>
      </link>
    </model>"""


def _flat_floor_model(surface: SurfaceProps = DEFAULT_SURFACE) -> str:
    return f"""
    <model name="flat_floor">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>50.0 50.0 0.04</size></box></geometry>
          <pose>0 0 -0.02 0 0 0</pose>
          {_surface_sdf(surface)}
        </collision>
        <visual name="visual">
          <geometry><box><size>50.0 50.0 0.04</size></box></geometry>
          <pose>0 0 -0.02 0 0 0</pose>
          <material><ambient>0.34 0.35 0.33 1</ambient><diffuse>0.42 0.43 0.40 1</diffuse></material>
        </visual>
      </link>
    </model>"""


# Gazebo's <gui><camera> looks along the camera's local +X axis (X-forward,
# Y-left, Z-up body frame — the same convention as <sensor type="camera">).
# pitch=90deg rotates that axis to world -Z (straight down); yaw=90deg then
# rotates the in-frame "up" direction to world +Y (north). The result: north
# up, east right — the same convention map_server/RViz use for the 2D map.
_GUI_TOP_DOWN_CAMERA = """
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>0 0 40 0 1.5708 1.5708</pose>
      </camera>
    </gui>
"""


def _world_sdf(name: str, body: str, *, include_gui_camera: bool = True) -> str:
    """Shared Gazebo world wrapper: plugin list, physics, sun light, optional top-down camera."""
    gui_block = _GUI_TOP_DOWN_CAMERA if include_gui_camera else ""
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="{name}">
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-forcetorque-system" name="ignition::gazebo::systems::ForceTorque"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
{gui_block}
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 12 0 0 0</pose>
      <diffuse>0.82 0.82 0.78 1</diffuse>
      <specular>0.18 0.18 0.16 1</specular>
      <direction>-0.45 0.20 -0.88</direction>
    </light>
{body}
  </world>
</sdf>
"""


def export_navigation_world_sdf(
    out_dir: Path,
    wall_rects: list[dict],
    *,
    wall_height: float,
    terrain_visual_obj: Path | None = None,
    wall_color: str = "0.42 0.43 0.41 1",
    surface: SurfaceProps = DEFAULT_SURFACE,
) -> Path:
    """Export the Nav2 demo world.

    Navigation worlds use a thick flat floor for stable skid-steer contact.
    The aligned height assets are still exported beside this world for later
    Gazebo/Isaac composition, but the height mesh is intentionally not included
    here.
    """

    world_path = out_dir / "world.sdf"
    models = []
    if terrain_visual_obj is not None:
        models.append(_terrain_visual_model(terrain_visual_obj))
    models.append(_flat_floor_model(surface=surface))
    for i, rect in enumerate(wall_rects):
        cx = float(rect["center"][0])
        cy = float(rect["center"][1])
        sx = float(rect["size"][0])
        sy = float(rect["size"][1])
        models.append(
            _box_model(
                f"nav_wall_{i:03d}",
                f"{cx:.3f} {cy:.3f} {wall_height / 2.0:.3f} 0 0 0",
                f"{sx:.3f} {sy:.3f} {wall_height:.3f}",
                wall_color,
                surface=surface,
            )
        )

    world_path.write_text(
        _world_sdf("generated_nav_maze", "".join(models), include_gui_camera=True),
        encoding="utf-8",
    )
    return world_path


def export_navigation_mesh_contact_world_sdf(
    out_dir: Path,
    terrain_sdf: Path,
    wall_rects: list[dict],
    *,
    wall_height: float,
    wall_color: str = "0.42 0.43 0.41 1",
    surface: SurfaceProps = DEFAULT_SURFACE,
) -> Path:
    """Export an experimental Nav/Gazebo world where wheels contact the mesh.

    Keep this separate from ``world.sdf``. Mesh contact is useful for terrain
    A/B tests, but the accepted Nav2 smoke baseline uses the thick flat floor
    world for repeatable skid-steer contact.
    """

    world_path = out_dir / "world_mesh_contact.sdf"
    terrain_uri = terrain_sdf.resolve().as_uri()
    models = [f"\n    <include><uri>{terrain_uri}</uri></include>"]
    for i, rect in enumerate(wall_rects):
        cx = float(rect["center"][0])
        cy = float(rect["center"][1])
        sx = float(rect["size"][0])
        sy = float(rect["size"][1])
        models.append(
            _box_model(
                f"nav_wall_{i:03d}",
                f"{cx:.3f} {cy:.3f} {wall_height / 2.0:.3f} 0 0 0",
                f"{sx:.3f} {sy:.3f} {wall_height:.3f}",
                wall_color,
                surface=surface,
            )
        )

    world_path.write_text(
        _world_sdf("generated_nav_maze_mesh_contact", "".join(models), include_gui_camera=True),
        encoding="utf-8",
    )
    return world_path


def export_world_sdf(
    out_dir: Path,
    terrain_sdf: Path,
    size_x: float,
    size_y: float,
    wall_height: float,
    wall_thickness: float,
    surface: SurfaceProps = DEFAULT_SURFACE,
    surround_copies: int = 0,
) -> Path:
    """surround_copies>0 tiles the same terrain.sdf model in a
    (2*surround_copies+1) x (2*surround_copies+1) grid around the center tile
    (1 -> 8 extra neighbors). Only meaningful for wall_height<=0 presets
    (e.g. rl_curriculum): without walls, a robot that strays past the
    nominal size_x/size_y tile drives straight into empty space and falls.
    Repeating the same heightmap tile means straying past that nominal
    boundary lands on more of the same terrain instead of a void, while the
    physical contact surface inside the original tile -- the thing a
    sim-to-sim eval actually cares about matching Isaac Lab -- is unchanged.
    """
    world_path = out_dir / "world.sdf"
    terrain_uri = terrain_sdf.resolve().as_uri()
    wall_color = "0.48 0.48 0.46 1"
    sx, sy, wh, wt = size_x, size_y, wall_height, wall_thickness
    models = []
    for i in range(-surround_copies, surround_copies + 1):
        for j in range(-surround_copies, surround_copies + 1):
            if i == 0 and j == 0:
                models.append(f"\n    <include><uri>{terrain_uri}</uri></include>")
            else:
                dx, dy = i * sx, j * sy
                models.append(
                    f"\n    <include><uri>{terrain_uri}</uri>"
                    f"<name>generated_heightmap_terrain_{i}_{j}</name>"
                    f"<pose>{dx:.3f} {dy:.3f} 0 0 0 0</pose></include>"
                )
    if wh > 0.0 and wt > 0.0:
        models.extend(
            [
                _box_model("boundary_north", f"0 {sy/2:.3f} {wh/2:.3f} 0 0 0", f"{sx:.3f} {wt:.3f} {wh:.3f}", wall_color, surface=surface),
                _box_model("boundary_south", f"0 {-sy/2:.3f} {wh/2:.3f} 0 0 0", f"{sx:.3f} {wt:.3f} {wh:.3f}", wall_color, surface=surface),
                _box_model("boundary_east", f"{sx/2:.3f} 0 {wh/2:.3f} 0 0 0", f"{wt:.3f} {sy:.3f} {wh:.3f}", wall_color, surface=surface),
                _box_model("boundary_west", f"{-sx/2:.3f} 0 {wh/2:.3f} 0 0 0", f"{wt:.3f} {sy:.3f} {wh:.3f}", wall_color, surface=surface),
            ]
        )
    world_path.write_text(
        _world_sdf("generated_baseline", "".join(models), include_gui_camera=True),
        encoding="utf-8",
    )
    return world_path
