import json
from pathlib import Path

import numpy as np


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
    _write_png(out_dir / "preview.png", height)
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


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


def export_terrain_sdf(out_dir: Path, obj_path: Path) -> Path:
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
        <surface>
          <friction><ode><mu>1.50</mu><mu2>1.50</mu2></ode></friction>
        </surface>
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


def _box_model(name: str, pose: str, size: str, color: str) -> str:
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>{size}</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{size}</size></box></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>
      </link>
    </model>"""


def export_world_sdf(
    out_dir: Path,
    terrain_sdf: Path,
    size_x: float,
    size_y: float,
    wall_height: float,
    wall_thickness: float,
) -> Path:
    world_path = out_dir / "world.sdf"
    terrain_uri = terrain_sdf.resolve().as_uri()
    wall_color = "0.48 0.48 0.46 1"
    sx = size_x
    sy = size_y
    wh = wall_height
    wt = wall_thickness
    models = []
    if wh > 0.0 and wt > 0.0:
        models.extend(
            [
                _box_model("boundary_north", f"0 {sy/2:.3f} {wh/2:.3f} 0 0 0", f"{sx:.3f} {wt:.3f} {wh:.3f}", wall_color),
                _box_model("boundary_south", f"0 {-sy/2:.3f} {wh/2:.3f} 0 0 0", f"{sx:.3f} {wt:.3f} {wh:.3f}", wall_color),
                _box_model("boundary_east", f"{sx/2:.3f} 0 {wh/2:.3f} 0 0 0", f"{wt:.3f} {sy:.3f} {wh:.3f}", wall_color),
                _box_model("boundary_west", f"{-sx/2:.3f} 0 {wh/2:.3f} 0 0 0", f"{wt:.3f} {sy:.3f} {wh:.3f}", wall_color),
            ]
        )
    world_path.write_text(
        f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="generated_baseline">
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-forcetorque-system" name="ignition::gazebo::systems::ForceTorque"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>

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

    <include><uri>{terrain_uri}</uri></include>
{''.join(models)}
  </world>
</sdf>
""",
        encoding="utf-8",
    )
    return world_path
