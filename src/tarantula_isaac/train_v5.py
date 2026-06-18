"""Active-suspension PPO training -- direct runner (no Hydra task registry needed).

Usage (from repo root, with isaac_venv active):
  python3 src/tarantula_isaac/train_v5.py [--num_envs 64] [--max_iterations 400]
                                           [--resume logs/rsl_rl/.../model_NNN.pt]
                                           [--headless]
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TERRAIN_DIR = REPO_ROOT / "generated" / "terrains" / "rl_curriculum" / "42"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Active-suspension PPO training")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=400)
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to warm-start from")
parser.add_argument(
    "--command-profile",
    choices=("mixed", "yaw_only", "stage0", "mission"),
    default="mixed",
    help="Command curriculum profile. stage0 uses low-speed primitive+mission commands; yaw_only samples pure turns only.",
)
parser.add_argument(
    "--command-resampling-time",
    type=float,
    default=None,
    help="Seconds between sampled cmd_vel commands during training.",
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
    "--action-rate-weight",
    type=float,
    default=None,
    help="Override action rate penalty weight.",
)
parser.add_argument(
    "--policy-init-std",
    type=float,
    default=None,
    help="Override PPO actor Gaussian initial std. Residual policies should start with small exploration.",
)
parser.add_argument(
    "--hip-action-target-limit",
    type=float,
    default=None,
    help="Active-suspension hip target clamp in rad.",
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
from isaacsim.core.utils.extensions import enable_extension
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
    enable_extension("isaacsim.asset.importer.urdf")
    ensure_tarantula_usd()

    env_cfg = TarantulaSuspensionEnvCfg()
    env_cfg.action_space = 6
    env_cfg.observation_space = 50
    if args.hip_action_target_limit is not None:
        env_cfg.hip_action_target_limit = float(args.hip_action_target_limit)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.terrain = make_shared_heightmap_terrain_cfg(
        args.terrain_dir,
        min_level=args.terrain_level_min,
        max_level=args.terrain_level_max,
    )
    if args.max_abs_wheel_omega is not None:
        env_cfg.max_abs_wheel_omega = float(args.max_abs_wheel_omega)
    if args.action_rate_weight is not None:
        env_cfg.reward_action_rate_weight = float(args.action_rate_weight)
    if args.command_resampling_time is not None:
        env_cfg.command_resampling_time_s = float(args.command_resampling_time)
    if args.command_profile == "yaw_only":
        env_cfg.command_stop_prob = 0.0
        env_cfg.command_straight_prob = 0.0
        env_cfg.command_pure_turn_prob = 1.0
        env_cfg.command_mission_prob = 0.0
        env_cfg.command_wz_range = (-0.25, 0.25)
        env_cfg.command_min_abs_wz = 0.25
    elif args.command_profile == "mission":
        env_cfg.command_stop_prob = 0.20
        env_cfg.command_straight_prob = 0.40
        env_cfg.command_pure_turn_prob = 0.40
        env_cfg.command_mission_prob = 0.70
        env_cfg.command_vx_range = (-0.16, 0.16)
        env_cfg.command_wz_range = (-0.25, 0.25)
        env_cfg.command_min_abs_vx = 0.08
        env_cfg.command_min_abs_wz = 0.12
    elif args.command_profile == "stage0":
        env_cfg.command_stop_prob = 0.20
        env_cfg.command_straight_prob = 0.40
        env_cfg.command_pure_turn_prob = 0.40
        env_cfg.command_mission_prob = 0.40
        env_cfg.command_vx_range = (-0.16, 0.16)
        env_cfg.command_wz_range = (-0.25, 0.25)
        env_cfg.command_min_abs_vx = 0.08
        env_cfg.command_min_abs_wz = 0.12
        env_cfg.obs_noise_std = min(float(env_cfg.obs_noise_std), 0.01)
        env_cfg.push_lin_vel_range = (-0.2, 0.2)

    agent_cfg = TarantulaSuspensionPPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations
    if args.max_iterations <= 5:
        agent_cfg.save_interval = 1
    if args.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = float(args.entropy_coef)
    if args.policy_init_std is not None:
        agent_cfg.actor.distribution_cfg.init_std = float(args.policy_init_std)
    agent_cfg.device = "cuda:0"
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _RSL_RL_VERSION)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    suffix = "_v5_active_suspension"
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + suffix
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
