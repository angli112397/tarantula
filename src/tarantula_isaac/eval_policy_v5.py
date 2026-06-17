"""Deterministic Isaac Lab rollout health check for Tarantula Stage B.

The script runs the same command sequence used by the Gazebo tracking checks
inside Isaac Lab before a policy is exported or judged in Gazebo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TERRAIN_DIR = REPO_ROOT / "generated" / "terrains" / "gazebo_demo" / "42"
DEFAULT_OUT = REPO_ROOT / "generated" / "benchmarks" / "isaac_eval" / "summary.json"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Tarantula Stage B policy in Isaac Lab.")
parser.add_argument("--terrain-dir", default=str(DEFAULT_TERRAIN_DIR))
parser.add_argument("--terrain-level-min", type=int, default=None)
parser.add_argument("--terrain-level-max", type=int, default=None)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--duration", type=float, default=3.0, help="Seconds per command segment.")
parser.add_argument(
    "--mode",
    choices=("zero", "open_loop", "npz"),
    default="open_loop",
    help="Action source: zero actions, analytic skid-steer baseline, or exported npz actor.",
)
parser.add_argument("--policy-npz", default="", help="Exported actor npz for --mode npz.")
parser.add_argument(
    "--open-loop-vx-scale",
    type=float,
    default=1.0,
    help="Scale cmd_vx before analytic skid-steer conversion in --mode open_loop.",
)
parser.add_argument(
    "--open-loop-wz-scale",
    type=float,
    default=1.0,
    help="Scale cmd_wz before analytic skid-steer conversion in --mode open_loop.",
)
parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON summary output path.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.importer.urdf")

from tarantula_control.rl_policy import RLWheelCompensationPolicy
from tarantula_isaac.robot import ensure_tarantula_usd
from tarantula_isaac.shared_heightmap_terrain import make_shared_heightmap_terrain_cfg
from tarantula_isaac.suspension_env import TarantulaSuspensionEnv, _quat_roll_pitch
from tarantula_isaac.suspension_env_cfg import TarantulaSuspensionEnvCfg


SEGMENTS = [
    ("stop", 0.0, 0.0),
    ("turn_left_from_drive_cmd", 0.1, 0.15),
    ("drive_after_left", 0.1, 0.0),
    ("turn_right_from_drive_cmd", 0.1, -0.15),
    ("drive_after_right", 0.1, 0.0),
    ("backward", -0.1, 0.0),
    ("turn_left_authority", 0.0, 0.25),
    ("turn_right_authority", 0.0, -0.25),
    ("final_stop", 0.0, 0.0),
]


def _reset(env: TarantulaSuspensionEnv) -> dict[str, torch.Tensor]:
    reset_out = env.reset()
    return reset_out[0] if isinstance(reset_out, tuple) else reset_out


def _open_loop_action(
    cfg: TarantulaSuspensionEnvCfg,
    vx: float,
    wz: float,
    num_envs: int,
    device: str,
    vx_scale: float,
    wz_scale: float,
) -> torch.Tensor:
    del vx, wz
    action = np.zeros(int(cfg.action_space), dtype=np.float32)
    action[:3] = np.asarray(
        [
            (wz_scale - 1.0) / float(cfg.track_scale_delta_limit),
            (vx_scale - 1.0) / float(cfg.drive_scale_delta_limit),
            (vx_scale - 1.0) / float(cfg.drive_scale_delta_limit),
        ],
        dtype=np.float32,
    )
    return torch.tensor(np.clip(action, -1.0, 1.0), device=device).repeat(num_envs, 1)


def _policy_action(policy: RLWheelCompensationPolicy, obs: dict[str, torch.Tensor], device: str) -> torch.Tensor:
    obs_np = obs["policy"].detach().cpu().numpy()
    action_np = np.stack([policy.act(row.astype(np.float32)) for row in obs_np], axis=0)
    return torch.tensor(action_np, dtype=torch.float32, device=device)


def _segment_summary(
    name: str,
    cmd_vx: float,
    cmd_wz: float,
    target_vx: float,
    target_wz: float,
    start_xy: torch.Tensor,
    rewards: list[torch.Tensor],
    actions: list[torch.Tensor],
    lin_vel_samples: list[torch.Tensor],
    ang_vel_samples: list[torch.Tensor],
    roll_samples: list[torch.Tensor],
    pitch_samples: list[torch.Tensor],
    terminations: dict[str, int],
    env: TarantulaSuspensionEnv,
) -> dict:
    end_xy = env._robot.data.root_pos_w[:, :2].detach()
    displacement = torch.linalg.norm(end_xy - start_xy, dim=-1)
    reward_tensor = torch.stack(rewards) if rewards else torch.zeros(1, env.num_envs, device=env.device)
    action_tensor = torch.cat(actions, dim=0) if actions else torch.zeros(1, env.cfg.action_space, device=env.device)
    lin_vel_tensor = (
        torch.stack(lin_vel_samples)
        if lin_vel_samples
        else env._robot.data.root_lin_vel_b.detach().unsqueeze(0)
    )
    ang_vel_tensor = (
        torch.stack(ang_vel_samples)
        if ang_vel_samples
        else env._robot.data.root_ang_vel_b.detach().unsqueeze(0)
    )
    roll_tensor = torch.stack(roll_samples) if roll_samples else torch.zeros(1, env.num_envs, device=env.device)
    pitch_tensor = torch.stack(pitch_samples) if pitch_samples else torch.zeros(1, env.num_envs, device=env.device)

    vx = lin_vel_tensor[:, :, 0]
    wz = ang_vel_tensor[:, :, 2]
    vx_err = vx - target_vx
    wz_err = wz - target_wz
    final_vx = lin_vel_tensor[-1, :, 0]
    final_wz = ang_vel_tensor[-1, :, 2]
    return {
        "segment": name,
        "cmd_vx": cmd_vx,
        "cmd_wz": cmd_wz,
        "target_vx": target_vx,
        "target_wz": target_wz,
        "mean_displacement_m": float(displacement.mean().item()),
        "mean_reward": float(reward_tensor.mean().item()),
        "mean_vx": float(vx.mean().item()),
        "mean_wz": float(wz.mean().item()),
        "final_vx": float(final_vx.mean().item()),
        "final_wz": float(final_wz.mean().item()),
        "rms_vx_error": float(torch.sqrt(torch.mean(torch.square(vx_err))).item()),
        "rms_wz_error": float(torch.sqrt(torch.mean(torch.square(wz_err))).item()),
        "max_abs_roll_rad": float(torch.max(torch.abs(roll_tensor)).item()),
        "max_abs_pitch_rad": float(torch.max(torch.abs(pitch_tensor)).item()),
        "action_saturation_rate": float((torch.abs(action_tensor) > 0.98).float().mean().item()),
        "termination_counts": terminations,
    }


def main() -> None:
    ensure_tarantula_usd()

    cfg = TarantulaSuspensionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.command_resampling_enabled = False
    cfg.terrain = make_shared_heightmap_terrain_cfg(
        args.terrain_dir,
        debug_vis=False,
        min_level=args.terrain_level_min,
        max_level=args.terrain_level_max,
    )

    policy = None
    if args.mode == "npz":
        if not args.policy_npz:
            raise ValueError("--policy-npz is required with --mode npz")
        policy = RLWheelCompensationPolicy(args.policy_npz)
        if policy.action_dim == 9:
            cfg.hip_action_enabled = True
            cfg.action_space = 9
            cfg.observation_space = 53
        if policy.action_dim != cfg.action_space:
            raise ValueError(
                f"policy action dim {policy.action_dim} does not match current Isaac env action space {cfg.action_space}"
            )
        cfg.max_abs_wheel_omega = float(policy.max_abs_wheel_omega)
        cfg.track_scale_delta_limit = float(policy.track_scale_delta_limit)
        cfg.drive_scale_delta_limit = float(policy.drive_scale_delta_limit)
        cfg.hip_action_target_limit = float(policy.hip_action_target_limit)

    env = TarantulaSuspensionEnv(cfg=cfg, render_mode=None)
    obs = _reset(env)

    term_terms = env._termination_terms()
    spawn_health = {
        "min_base_z": float(env._robot.data.root_pos_w[:, 2].min().item()),
        "max_base_z": float(env._robot.data.root_pos_w[:, 2].max().item()),
        "initial_termination_counts": {
            key: int(torch.count_nonzero(value).item())
            for key, value in term_terms.items()
        },
    }

    dt = float(cfg.sim.dt) * float(cfg.decimation)
    steps_per_segment = max(1, int(round(args.duration / dt)))
    segments = []

    with torch.inference_mode():
        for name, cmd_vx, cmd_wz in SEGMENTS:
            env._cmd_vx[:] = cmd_vx
            env._cmd_wz[:] = cmd_wz
            env._update_execution_commands()
            target_vx = float(env._exec_cmd_vx.mean().item())
            target_wz = float(env._exec_cmd_wz.mean().item())
            start_xy = env._robot.data.root_pos_w[:, :2].detach().clone()
            rewards = []
            actions = []
            lin_vel_samples = []
            ang_vel_samples = []
            roll_samples = []
            pitch_samples = []
            terminations = {key: 0 for key in env._termination_terms().keys()}
            terminations["time_out"] = 0

            for _ in range(steps_per_segment):
                env._cmd_vx[:] = cmd_vx
                env._cmd_wz[:] = cmd_wz
                if args.mode == "zero":
                    action = torch.zeros((env.num_envs, cfg.action_space), device=env.device)
                elif args.mode == "open_loop":
                    action = _open_loop_action(
                        cfg,
                        cmd_vx,
                        cmd_wz,
                        env.num_envs,
                        env.device,
                        args.open_loop_vx_scale,
                        args.open_loop_wz_scale,
                    )
                else:
                    action = _policy_action(policy, obs, env.device)

                obs, reward, terminated, timeout, extras = env.step(action)
                rewards.append(reward.detach())
                actions.append(action.detach())
                lin_vel_samples.append(env._robot.data.root_lin_vel_b.detach().clone())
                ang_vel_samples.append(env._robot.data.root_ang_vel_b.detach().clone())
                roll, pitch = _quat_roll_pitch(env._imu.data.quat_w.detach())
                roll_samples.append(roll.detach().clone())
                pitch_samples.append(pitch.detach().clone())
                log = extras.get("log", {}) if isinstance(extras, dict) else {}
                for key in env._termination_terms().keys():
                    terminations[key] += int(log.get(f"Episode_Termination/{key}", 0))
                terminations["time_out"] += int(log.get("Episode_Termination/time_out", 0))
                terminations["any"] = terminations.get("any", 0) + int(torch.count_nonzero(terminated).item())
                terminations["any_time_out"] = terminations.get("any_time_out", 0) + int(torch.count_nonzero(timeout).item())

            segments.append(
                _segment_summary(
                    name,
                    cmd_vx,
                    cmd_wz,
                    target_vx,
                    target_wz,
                    start_xy,
                    rewards,
                    actions,
                    lin_vel_samples,
                    ang_vel_samples,
                    roll_samples,
                    pitch_samples,
                    terminations,
                    env,
                )
            )

    summary = {
        "mode": args.mode,
        "policy_npz": args.policy_npz,
        "terrain_dir": args.terrain_dir,
        "terrain_level_min": args.terrain_level_min,
        "terrain_level_max": args.terrain_level_max,
        "num_envs": args.num_envs,
        "duration_s": args.duration,
        "open_loop_vx_scale": args.open_loop_vx_scale,
        "open_loop_wz_scale": args.open_loop_wz_scale,
        "track_scale_delta_limit": cfg.track_scale_delta_limit,
        "drive_scale_delta_limit": cfg.drive_scale_delta_limit,
        "max_abs_wheel_omega": cfg.max_abs_wheel_omega,
        "steps_per_segment": steps_per_segment,
        "spawn_health": spawn_health,
        "segments": segments,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
