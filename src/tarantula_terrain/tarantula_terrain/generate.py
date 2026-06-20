import argparse
from pathlib import Path

from .exporters import export_height_assets, export_obj, export_terrain_sdf, export_world_sdf
from .generator import generate_heightmap
from .terrain_cfg import PRESETS


def generate(preset: str, seed: int, output_root: Path) -> Path:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset '{preset}', available: {', '.join(sorted(PRESETS))}")

    cfg = PRESETS[preset]
    out_dir = output_root / preset / str(seed)
    height, metadata = generate_heightmap(cfg, seed)
    # Unwalled presets (rl_curriculum) have no physical barrier at size_x/size_y,
    # so world.sdf tiles the same heightmap around the center tile to avoid a
    # void at that boundary -- see export_world_sdf's docstring. Walled presets
    # (gazebo_demo) already contain the robot, so leave them untiled.
    surround_copies = 1 if cfg.wall_height <= 0.0 and cfg.wall_thickness <= 0.0 else 0
    metadata["surround_copies"] = surround_copies
    export_height_assets(out_dir, height, metadata)
    obj_path = export_obj(out_dir, height, cfg.resolution)
    terrain_sdf = export_terrain_sdf(out_dir, obj_path)
    export_world_sdf(
        out_dir,
        terrain_sdf,
        cfg.size_x,
        cfg.size_y,
        cfg.wall_height,
        cfg.wall_thickness,
        surround_copies=surround_copies,
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Tarantula heightmap terrain assets.")
    parser.add_argument("--preset", default="gazebo_demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        default="generated/terrains",
        help="Output root directory. Assets are written to <root>/<preset>/<seed>/",
    )
    args = parser.parse_args()
    out_dir = generate(args.preset, args.seed, Path(args.output_root))
    print(out_dir)


if __name__ == "__main__":
    main()
