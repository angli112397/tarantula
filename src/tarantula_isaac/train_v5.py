"""Active-suspension PPO training -- direct runner (no Hydra task registry needed).

Usage (from repo root, with isaac_venv active):
  python3 src/tarantula_isaac/train_v5.py [--num_envs 64] [--max_iterations 400]
                                           [--resume logs/rsl_rl/.../model_NNN.pt]
                                           [--headless]
"""

import argparse
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
    "--pursuit-prob",
    type=float,
    default=None,
    help="Opt in to pure-pursuit checkpoint-chasing commands (CommandsCfg.pursuit_prob defaults to 0.0).",
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


# CommandsCfg field overrides per --command-profile. "mixed" (the default) is
# the CommandsCfg() baseline as-is, so it has no entry here.
COMMAND_PROFILES: dict[str, dict] = {
    "yaw_only": dict(
        stop_prob=0.0, straight_prob=0.0, turn_prob=1.0, curve_prob=0.0, mission_prob=0.0,
        wz_range=(-0.25, 0.25), min_abs_wz=0.25,
    ),
    "mission": dict(
        stop_prob=0.20, straight_prob=0.40, turn_prob=0.20, curve_prob=0.20, mission_prob=0.70,
        vx_range=(-0.16, 0.16), wz_range=(-0.25, 0.25), min_abs_vx=0.08, min_abs_wz=0.12,
    ),
    "stage0": dict(
        stop_prob=0.20, straight_prob=0.40, turn_prob=0.20, curve_prob=0.20, mission_prob=0.40,
        vx_range=(-0.16, 0.16), wz_range=(-0.25, 0.25), min_abs_vx=0.08, min_abs_wz=0.12,
    ),
}


def main():
    enable_extension("isaacsim.asset.importer.urdf")
    ensure_tarantula_usd()

    env_cfg = TarantulaSuspensionEnvCfg()
    env_cfg.action_space = 6
    env_cfg.observation_space = 56
    if args.hip_action_target_limit is not None:
        env_cfg.hip_action_target_limit = float(args.hip_action_target_limit)
    env_cfg.scene.num_envs = args.num_envs
    # No min_level/max_level override: training always gets the terrain's
    # full difficulty range (see suspension_env.py's _reset_idx, which
    # re-rolls a random tile from that full range on every reset).
    env_cfg.terrain = make_shared_heightmap_terrain_cfg(args.terrain_dir)
    if args.max_abs_wheel_omega is not None:
        env_cfg.max_abs_wheel_omega = float(args.max_abs_wheel_omega)
    if args.command_resampling_time is not None:
        env_cfg.commands.resampling_time_s = float(args.command_resampling_time)
    if args.command_profile in COMMAND_PROFILES:
        env_cfg.commands = env_cfg.commands.replace(**COMMAND_PROFILES[args.command_profile])
    if args.command_profile == "stage0":
        env_cfg.domain_rand.obs_noise_std = min(float(env_cfg.domain_rand.obs_noise_std), 0.01)
        # Opt in to push perturbation DR per DomainRandCfg's docstring in
        # suspension_env_cfg.py: stage0 trains after the deterministic
        # baseline already proves stable posture control, so it's safe to
        # add random pushes here without confusing them with contact bugs.
        env_cfg.domain_rand.push_interval_steps = (150, 300)
        env_cfg.domain_rand.push_lin_vel_range = (-0.2, 0.2)
    if args.pursuit_prob is not None:
        env_cfg.commands = env_cfg.commands.replace(pursuit_prob=float(args.pursuit_prob))

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
