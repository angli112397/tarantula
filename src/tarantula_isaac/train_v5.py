"""M7 v5 PPO training — direct runner (no Hydra task registry needed).

Usage (from repo root, with isaac_venv active):
  python3 src/tarantula_isaac/train_v5.py [--num_envs 64] [--max_iterations 400]
                                           [--resume logs/rsl_rl/.../model_NNN.pt]
                                           [--headless]
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TERRAIN_DIR = REPO_ROOT / "generated" / "terrains" / "gazebo_demo" / "42"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="M7 v5 PPO training")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=400)
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to warm-start from")
parser.add_argument(
    "--command-profile",
    choices=("mixed", "yaw_only"),
    default="mixed",
    help="Command curriculum profile. yaw_only samples pure turns only.",
)
parser.add_argument(
    "--track-scale-delta-limit",
    type=float,
    default=None,
    help="Override maximum fractional effective-track correction.",
)
parser.add_argument(
    "--drive-scale-delta-limit",
    type=float,
    default=None,
    help="Override maximum fractional left/right drive correction.",
)
parser.add_argument(
    "--max-abs-wheel-omega",
    type=float,
    default=None,
    help="Override final wheel target clamp in rad/s.",
)
parser.add_argument(
    "--entropy-coef",
    type=float,
    default=None,
    help="Override PPO entropy coefficient.",
)
parser.add_argument(
    "--action-saturation-weight",
    type=float,
    default=None,
    help="Override action saturation penalty weight.",
)
parser.add_argument(
    "--terrain-dir",
    default=str(DEFAULT_TERRAIN_DIR),
    help="Generated terrain directory containing height.npy and metadata.json.",
)
parser.add_argument(
    "--terrain-level-min",
    type=int,
    default=None,
    help="Minimum rl_curriculum terrain row to sample for resets.",
)
parser.add_argument(
    "--terrain-level-max",
    type=int,
    default=None,
    help="Maximum rl_curriculum terrain row to sample for resets.",
)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True  # always headless for training
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- post-AppLauncher imports (pxr / isaaclab internals now available) ----
import os
import time
from datetime import datetime

import torch
from rsl_rl.runners import OnPolicyRunner

import importlib.metadata as _meta
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaaclab.utils.io import dump_yaml

_RSL_RL_VERSION = _meta.version("rsl-rl-lib")

# Our env + cfg — imports fine after AppLauncher
from tarantula_isaac.suspension_env import TarantulaSuspensionEnv
from tarantula_isaac.suspension_env_cfg import TarantulaSuspensionEnvCfg
from tarantula_isaac.shared_heightmap_terrain import make_shared_heightmap_terrain_cfg
from tarantula_isaac.agents.rsl_rl_ppo_cfg import TarantulaSuspensionPPORunnerCfg
from tarantula_isaac.robot import ensure_tarantula_usd

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    ensure_tarantula_usd()

    env_cfg = TarantulaSuspensionEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.terrain = make_shared_heightmap_terrain_cfg(
        args.terrain_dir,
        min_level=args.terrain_level_min,
        max_level=args.terrain_level_max,
    )
    if args.track_scale_delta_limit is not None:
        env_cfg.track_scale_delta_limit = float(args.track_scale_delta_limit)
    if args.drive_scale_delta_limit is not None:
        env_cfg.drive_scale_delta_limit = float(args.drive_scale_delta_limit)
    if args.max_abs_wheel_omega is not None:
        env_cfg.max_abs_wheel_omega = float(args.max_abs_wheel_omega)
    if args.action_saturation_weight is not None:
        env_cfg.reward_action_saturation_weight = float(args.action_saturation_weight)
    if args.command_profile == "yaw_only":
        env_cfg.command_stop_prob = 0.0
        env_cfg.command_straight_prob = 0.0
        env_cfg.command_pure_turn_prob = 1.0
        env_cfg.command_wz_range = (-0.25, 0.25)
        env_cfg.command_min_abs_wz = 0.25

    agent_cfg = TarantulaSuspensionPPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations
    if args.max_iterations <= 5:
        agent_cfg.save_interval = 1
    if args.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = float(args.entropy_coef)
    agent_cfg.device = "cuda:0"
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _RSL_RL_VERSION)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_v5_stage_a_wheel_only"
    log_dir = os.path.join(log_root, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    env = TarantulaSuspensionEnv(cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        runner.load(args.resume)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    start = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[INFO] Training finished in {round(time.time()-start,1)}s")
    print(f"[INFO] Checkpoints saved in: {log_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
