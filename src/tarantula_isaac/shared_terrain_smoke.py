"""Headless smoke check for the shared Gazebo/Isaac heightmap terrain."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TERRAIN_DIR = REPO_ROOT / "generated" / "terrains" / "gazebo_demo" / "42"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test shared Tarantula heightmap terrain in Isaac Lab.")
parser.add_argument("--terrain-dir", default=str(DEFAULT_TERRAIN_DIR))
parser.add_argument("--num-envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.importer.urdf")
from tarantula_isaac.robot import ensure_tarantula_usd
from tarantula_isaac.shared_heightmap_terrain import make_shared_heightmap_terrain_cfg
from tarantula_isaac.suspension_env import TarantulaSuspensionEnv
from tarantula_isaac.suspension_env_cfg import TarantulaSuspensionEnvCfg


def main() -> None:
    ensure_tarantula_usd()
    cfg = TarantulaSuspensionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.command_resampling_enabled = False
    cfg.terrain = make_shared_heightmap_terrain_cfg(args.terrain_dir, debug_vis=False)

    env = TarantulaSuspensionEnv(cfg=cfg, render_mode=None)
    print(f"TERRAIN_IMPORTER={type(env._terrain).__name__}")
    print(f"ENV_ORIGINS_SHAPE={tuple(env._terrain.env_origins.shape)}")
    print(f"ENV_ORIGINS={env._terrain.env_origins[:args.num_envs].cpu().numpy().round(3).tolist()}")

    actions = torch.zeros((args.num_envs, cfg.action_space), device=env.device)
    env.step(actions)
    env.close()
    print("SHARED_TERRAIN_SMOKE_OK")


if __name__ == "__main__":
    main()
    simulation_app.close()
