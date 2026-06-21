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
# (this bit us four times now: once for gazebo_demo/42's contact surface,
# once for rl_curriculum's unsmoothed inner tile-edge cliffs, once for its
# unflattened outer-edge surround_copies seam, once for the world-origin
# spawn flattening that couldn't fully smooth a 4-tile junction -- see
# _generate_rl_curriculum, generator._taper_outer_edge, and
# generate_heightmap's comment on removing _clear_spawn).
# 3->4: export_obj now writes vt (UV) coords + an elevation colormap texture;
# dirs generated before this have no terrain_colormap.png and render flat.
# 4->5: generator.py tapers height to 0 within edge_taper_band of the outer
# rectangle so surround_copies' repeat seam is flat on both sides; dirs
# generated before this can have a multi-cm cliff exactly at that seam.
# 5->6: removed _clear_spawn's global-origin flattening entirely (see
# generate_heightmap) -- dirs generated before this still have the smoothed-
# but-still-20-30deg blend ring around world (0,0); the deployable fix is
# spawning on a tile's own flat _clear_platform square instead (sim.launch.py
# spawn_x/spawn_y), not anything encoded in the heightmap itself.
# 6->7: export_obj now alternates each cell's diagonal split direction
# (zigzag) instead of always cutting the same way -- the old uniform
# direction bakes a fixed-orientation ridge into the collision surface
# across the whole terrain (see gazebo-classic#2838/dart#1069 on OGRE vs
# ODE/DART triangulation conventions). Investigated as a candidate cause of
# this project's direction-dependent traction symptoms; empirically it did
# not fix the underlying skid-steer-on-mesh issue (a deeper, still-open
# dartsim/gz-physics5 limitation -- primitives turn correctly, mesh/
# heightmap collision does not, regardless of triangulation), but the old
# uniform-direction triangulation was a real bug independent of that and is
# worth keeping fixed. Dirs generated before this have the old, biased
# triangulation.
# 7->8: DEFAULT_SURFACE.kd 10.0->140.0, matching tarantula_v3.urdf.xacro's
# wheel <kd> -- found mismatched with no comment justifying a deliberate
# difference during a full contact-parameter audit. Dirs generated before
# this have the unmatched terrain-side kd.
GENERATOR_SCHEMA_VERSION = 8


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


# Elevation colormap for the terrain mesh's diffuse texture: dark
# soil-brown at the lowest point through warm tan to pale sun-bleached sand
# at the highest. Deliberately not matplotlib's "terrain"/"gist_earth" --
# those put blue at the low end (built for real topography that bottoms out
# at sea level), which would look like water pooling in this project's pits
# and ditches. Stops chosen for contrast on a rock/dirt heightfield, not
# geographic accuracy.
_ELEVATION_COLOR_STOPS = (
    (0.00, (0.22, 0.19, 0.15)),
    (0.50, (0.55, 0.45, 0.32)),
    (1.00, (0.86, 0.81, 0.69)),
)


def _write_elevation_colormap(path: Path, width: int = 256, height: int = 8) -> None:
    from PIL import Image

    stops_t = np.array([s[0] for s in _ELEVATION_COLOR_STOPS])
    stops_rgb = np.array([s[1] for s in _ELEVATION_COLOR_STOPS])
    t = np.linspace(0.0, 1.0, width)
    row = np.stack([np.interp(t, stops_t, stops_rgb[:, c]) for c in range(3)], axis=-1)
    row = (row * 255.0).clip(0, 255).astype(np.uint8)
    img = np.tile(row, (height, 1, 1))
    Image.fromarray(img, mode="RGB").save(path)


def export_obj(out_dir: Path, height: np.ndarray, resolution: float) -> Path:
    obj_path = out_dir / "terrain.obj"
    mtl_path = out_dir / "terrain.mtl"
    colormap_path = out_dir / "terrain_colormap.png"
    ny, nx = height.shape
    x0 = -(nx - 1) * resolution / 2.0
    y0 = -(ny - 1) * resolution / 2.0
    gy, gx = np.gradient(height, resolution, resolution)
    normals = np.dstack((-gx, -gy, np.ones_like(height)))
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(norms, 1e-9)

    # Same min/max normalization convention as _write_png, so the mesh's
    # color banding lines up with any height.npy preview/debug image.
    h_lo = float(height.min())
    h_span = max(float(height.max()) - h_lo, 1e-6)

    _write_elevation_colormap(colormap_path)

    with mtl_path.open("w", encoding="ascii") as f:
        f.write("newmtl terrain_mat\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.05 0.05 0.05\n")
        f.write(f"map_Kd {colormap_path.name}\n")

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
                u = (height[iy, ix] - h_lo) / h_span
                f.write(f"vt {u:.6f} 0.5\n")
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
                # Zigzag (alternate which corner-pair the diagonal connects,
                # by cell parity) instead of always splitting v0-v3: ODE/DART
                # without bullet's collision detector always uses the same
                # diagonal direction for every cell, baking a fixed-direction
                # ridge into the collision surface across the whole terrain
                # (confirmed via gazebo-classic#2838/dart#1069 -- matching
                # OGRE's own zigzag convention needs this explicitly, DART
                # only does it automatically when paired with bullet's
                # collision detector). A uniform diagonal direction is a
                # plausible contributor to this project's direction-dependent
                # traction symptoms on mesh collision.
                if (ix + iy) % 2 == 0:
                    f.write(f"f {v0}/{v0}/{v0} {v1}/{v1}/{v1} {v3}/{v3}/{v3}\n")
                    f.write(f"f {v0}/{v0}/{v0} {v3}/{v3}/{v3} {v2}/{v2}/{v2}\n")
                else:
                    f.write(f"f {v0}/{v0}/{v0} {v1}/{v1}/{v1} {v2}/{v2}/{v2}\n")
                    f.write(f"f {v1}/{v1}/{v1} {v3}/{v3}/{v3} {v2}/{v2}/{v2}\n")
    return obj_path


@dataclass(frozen=True)
class SurfaceProps:
    """Friction/contact tuning shared by every collision geometry we export.

    These are isotropic (mu == mu2, no fdir1), so the SDF <ode> element
    name is just SDFormat's generic tag namespace for contact-compliance
    parameters, not a statement about which engine reads them -- the world
    actually runs on dartsim (see _world_sdf's physics type comment), and
    an isotropic friction value has no direction to get wrong, so it reads
    correctly under dartsim with no extra attributes needed. Without
    slip/contact damping, defaults leave thin boxes and meshes essentially
    frictionless (skid-steer wheels can't generate a yaw moment), so this
    is threaded through every exporter instead of being a one-off block on
    a single geometry type.
    """

    mu: float = 1.50
    mu2: float = 1.50
    slip1: float = 0.001
    slip2: float = 0.001
    kp: float = 1_000_000.0
    # Matches tarantula_v3.urdf.xacro's wheel <kd> -- found mismatched at
    # 10.0 vs the wheel's 140.0 during a full-codebase contact-parameter
    # audit, no comment on either side explaining a deliberate difference.
    # Unified to the wheel's value since it looks like the one actually
    # tuned (140.0 isn't a generic placeholder the way 10.0 is).
    kd: float = 140.0
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
        <!-- No <material> override here: terrain.obj's own MTL now carries
             an elevation colormap texture (see export_obj). An SDF-level
             <material> block on a <visual> replaces whatever material the
             mesh file specifies, same gotcha as the GUI plugin merge issue
             documented on _GUI_TOP_DOWN_CAMERA: it would silently flatten
             the mesh back to one solid color. -->
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


# A world that declares its own <gui> with ANY <plugin> child replaces
# Gazebo's entire default GUI plugin set (~/.ignition/gazebo/6/gui.config)
# instead of merging with it -- confirmed empirically: adding just a bare
# <camera> tag silently dropped MinimalScene/CameraTracking too, so
# /gui/follow and /gui/follow/offset (chase-cam control) disappeared along
# with the 3D view itself. So this block re-declares the full default
# plugin set (MinimalScene's camera_pose replaces the old bare <camera> tag).
#
# MinimalScene's camera_pose: Gazebo's camera looks along its local +X axis
# (X-forward, Y-left, Z-up). pitch=90deg rotates that axis to world -Z
# (straight down); yaw=90deg then rotates in-frame "up" to world +Y (north)
# -- north up, east right, the same convention map_server/RViz use.
#
# Demo-recording chase cam: /gui/follow (StringMsg model name, e.g.
# "tarantula") + /gui/follow/offset (Vector3d, applied in the followed
# model's local/yaw-rotating frame -- confirmed empirically the camera
# auto-aims at the target continuously as it moves, i.e. a true chase
# camera, not a fixed-offset translation) -- call these once via `ign
# service` after launch, no extra plugin or script needed. Gazebo's own
# VideoRecorder GUI plugin was tried for capture-side automation, but its
# /gui/record_video service never advertises on this machine, even running
# Gazebo's own stock recording-tutorial world unmodified -- looks like an
# environment/EGL issue, not something fixable from the SDF. Record the GUI
# window with a normal screen recorder instead.
_GUI_TOP_DOWN_CAMERA = """
    <gui fullscreen="0">
      <plugin filename="MinimalScene" name="3D View">
        <ignition-gui>
          <title>3D View</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="string" key="state">docked</property>
        </ignition-gui>
        <engine>ogre2</engine>
        <scene>scene</scene>
        <ambient_light>0.4 0.4 0.4</ambient_light>
        <background_color>0.8 0.8 0.8</background_color>
        <camera_pose>0 0 40 0 1.5708 1.5708</camera_pose>
      </plugin>
      <plugin filename="InteractiveViewControl" name="Interactive view control">
        <ignition-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </ignition-gui>
      </plugin>
      <plugin filename="CameraTracking" name="Camera Tracking">
        <ignition-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </ignition-gui>
      </plugin>
      <plugin filename="GzSceneManager" name="Scene Manager">
        <ignition-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </ignition-gui>
      </plugin>
      <plugin filename="SelectEntities" name="Select Entities">
        <ignition-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </ignition-gui>
      </plugin>
      <plugin filename="VisualizationCapabilities" name="Visualization Capabilities">
        <ignition-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </ignition-gui>
      </plugin>
      <plugin filename="WorldControl" name="World control">
        <ignition-gui>
          <title>World control</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">72</property>
          <property type="double" key="width">121</property>
          <property type="double" key="z">1</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View">
            <line own="left" target="left"/>
            <line own="bottom" target="bottom"/>
          </anchors>
        </ignition-gui>
        <play_pause>true</play_pause>
        <step>true</step>
        <start_paused>false</start_paused>
        <use_event>true</use_event>
      </plugin>
      <plugin filename="WorldStats" name="World stats">
        <ignition-gui>
          <title>World stats</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">110</property>
          <property type="double" key="width">290</property>
          <property type="double" key="z">1</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View">
            <line own="right" target="right"/>
            <line own="bottom" target="bottom"/>
          </anchors>
        </ignition-gui>
        <sim_time>true</sim_time>
        <real_time>true</real_time>
        <real_time_factor>true</real_time_factor>
        <iterations>true</iterations>
      </plugin>
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
    <!-- "ode" was never real: no libignition-physics-*-ode-plugin exists on
         this install (only bullet/dartsim/tpe do), so gz-sim's Physics
         system silently fell back to its own default (dartsim) the entire
         time this said "ode", confirmed via -v 4 startup log ("Loaded
         [ignition::physics::dartsim::Plugin]"). Declaring it explicitly so
         the SDF says what's actually running, and because correctly fixing
         the wheel's anisotropic friction (tarantula_v3.urdf.xacro's fdir1)
         requires knowing for certain which engine: DART's directional
         friction needs the gz:expressed_in attribute, which has no ODE
         equivalent. -->
    <physics type="dartsim">
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
