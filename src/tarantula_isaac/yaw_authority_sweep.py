"""Sweep Isaac yaw authority without training.

This script answers whether the current wheel model and structured-compensation
envelope can physically produce commanded yaw before spending more PPO time.
It runs pure-turn open-loop trials with configurable track-scale corrections
and wheel limits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TERRAIN_DIR = REPO_ROOT / "generated" / "terrains" / "gazebo_demo" / "42"
DEFAULT_OUT = REPO_ROOT / "generated" / "benchmarks" / "isaac_eval" / "yaw_authority_sweep.json"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep Tarantula Stage A yaw authority in Isaac Lab.")
parser.add_argument("--terrain-dir", default=str(DEFAULT_TERRAIN_DIR))
parser.add_argument("--terrain-level-min", type=int, default=None)
parser.add_argument("--terrain-level-max", type=int, default=None)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--duration", type=float, default=0.8, help="Seconds per pure-turn trial.")
parser.add_argument("--cmd-wz", default="0.25", help="Comma-separated yaw commands in rad/s.")
parser.add_argument("--cmd-vx", type=float, default=0.0, help="Forward velocity during yaw trials.")
parser.add_argument("--max-wheel-omegas", default="6.0,8.0,10.0")
parser.add_argument("--track-actions", default="0.0,0.5,1.0")
parser.add_argument("--out", default=str(DEFAULT_OUT))
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.importer.urdf")

from tarantula_control.control_interfaces import EFFECTIVE_TRACK, WHEEL_DIRECTION, WHEEL_RADIUS
from tarantula_control.suspension_core import LEGS
from tarantula_isaac.robot import ensure_tarantula_usd
from tarantula_isaac.shared_heightmap_terrain import make_shared_heightmap_terrain_cfg
from tarantula_isaac.suspension_env import TarantulaSuspensionEnv, _quat_roll_pitch
from tarantula_isaac.suspension_env_cfg import TarantulaSuspensionEnvCfg


def _parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _reset(env: TarantulaSuspensionEnv) -> dict[str, torch.Tensor]:
    reset_out = env.reset()
    return reset_out[0] if isinstance(reset_out, tuple) else reset_out


def _open_loop_action(
    track_action: float,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    action = torch.zeros(3, dtype=torch.float32, device=device)
    action[0] = max(-1.0, min(1.0, float(track_action)))
    return action.repeat(num_envs, 1)


def _structured_wheel_target(
    cfg: TarantulaSuspensionEnvCfg,
    cmd_vx: float,
    cmd_wz: float,
    track_action: float,
    max_wheel_omega: float,
) -> list[float]:
    track_delta = max(-1.0, min(1.0, float(track_action))) * float(cfg.track_scale_delta_limit)
    vx_fraction = min(abs(cmd_vx) / max(float(cfg.track_scale_transition_vx), 1.0e-6), 1.0)
    if abs(cmd_wz) < 1.0e-4:
        base_track_scale = float(cfg.arc_track_scale)
    else:
        base_track_scale = (
            float(cfg.arc_track_scale) * vx_fraction
            + float(cfg.pure_turn_track_scale) * (1.0 - vx_fraction)
        )
    turn_track = EFFECTIVE_TRACK * base_track_scale * (1.0 + track_delta)
    left = (cmd_vx - 0.5 * turn_track * cmd_wz) / WHEEL_RADIUS
    right = (cmd_vx + 0.5 * turn_track * cmd_wz) / WHEEL_RADIUS
    direction = [WHEEL_DIRECTION[leg] for leg in LEGS]
    raw = [left, right, left, right, left, right]
    return [max(-max_wheel_omega, min(max_wheel_omega, raw[i] * direction[i])) for i in range(6)]


def _termination_counts(env: TarantulaSuspensionEnv) -> dict[str, int]:
    return {key: int(torch.count_nonzero(value).item()) for key, value in env._termination_terms().items()}


def _run_trial(
    env: TarantulaSuspensionEnv,
    *,
    cmd_vx: float,
    cmd_wz: float,
    max_wheel_omega: float,
    track_action: float,
    steps: int,
) -> dict:
    env.cfg.max_abs_wheel_omega = float(max_wheel_omega)
    obs = _reset(env)
    del obs

    env._cmd_vx[:] = cmd_vx
    env._cmd_wz[:] = cmd_wz
    action = _open_loop_action(track_action, env.num_envs, env.device)
    wz_samples = []
    vx_samples = []
    roll_samples = []
    pitch_samples = []
    terminations = {key: 0 for key in env._termination_terms()}
    terminations["any"] = 0
    terminations["time_out"] = 0

    # Use no_grad instead of inference_mode: Isaac Lab updates articulation
    # buffers in-place across resets, and inference tensors reject that update.
    with torch.no_grad():
        for _ in range(steps):
            env._cmd_vx[:] = cmd_vx
            env._cmd_wz[:] = cmd_wz
            _, _, terminated, timeout, extras = env.step(action)
            lin_vel_b = env._robot.data.root_lin_vel_b.detach().clone()
            ang_vel_b = env._robot.data.root_ang_vel_b.detach().clone()
            roll, pitch = _quat_roll_pitch(env._imu.data.quat_w.detach())
            vx_samples.append(lin_vel_b[:, 0])
            wz_samples.append(ang_vel_b[:, 2])
            roll_samples.append(roll.detach().clone())
            pitch_samples.append(pitch.detach().clone())
            log = extras.get("log", {}) if isinstance(extras, dict) else {}
            for key in env._termination_terms().keys():
                terminations[key] += int(log.get(f"Episode_Termination/{key}", 0))
            terminations["any"] += int(torch.count_nonzero(terminated).item())
            terminations["time_out"] += int(torch.count_nonzero(timeout).item())

    vx = torch.stack(vx_samples)
    wz = torch.stack(wz_samples)
    roll = torch.stack(roll_samples)
    pitch = torch.stack(pitch_samples)
    target = torch.full_like(wz, cmd_wz)
    final_target = _structured_wheel_target(env.cfg, cmd_vx, cmd_wz, track_action, max_wheel_omega)
    return {
        "cmd_vx": cmd_vx,
        "cmd_wz": cmd_wz,
        "max_abs_wheel_omega": max_wheel_omega,
        "track_action": track_action,
        "track_scale_delta": track_action * env.cfg.track_scale_delta_limit,
        "action_saturation_rate": float((torch.abs(action) > 0.98).float().mean().item()),
        "final_wheel_targets": [float(v) for v in final_target],
        "mean_vx": float(vx.mean().item()),
        "mean_wz": float(wz.mean().item()),
        "final_vx": float(vx[-1].mean().item()),
        "final_wz": float(wz[-1].mean().item()),
        "rms_wz_error": float(torch.sqrt(torch.mean(torch.square(wz - target))).item()),
        "yaw_ratio": float((wz.mean() / cmd_wz).item()) if abs(cmd_wz) > 1.0e-6 else 0.0,
        "max_abs_roll_rad": float(torch.max(torch.abs(roll)).item()),
        "max_abs_pitch_rad": float(torch.max(torch.abs(pitch)).item()),
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
    env = TarantulaSuspensionEnv(cfg=cfg, render_mode=None)
    _reset(env)

    cmd_wz_values = _parse_floats(args.cmd_wz)
    max_wheel_omegas = _parse_floats(args.max_wheel_omegas)
    track_actions = _parse_floats(args.track_actions)
    dt = float(cfg.sim.dt) * float(cfg.decimation)
    steps = max(1, int(round(args.duration / dt)))

    trials = []
    for cmd_wz_mag in cmd_wz_values:
        for direction in (1.0, -1.0):
            cmd_wz = direction * abs(cmd_wz_mag)
            for max_wheel_omega in max_wheel_omegas:
                for track_action in track_actions:
                    trials.append(
                        _run_trial(
                            env,
                            cmd_vx=args.cmd_vx,
                            cmd_wz=cmd_wz,
                            max_wheel_omega=max_wheel_omega,
                            track_action=track_action,
                            steps=steps,
                        )
                    )

    summary = {
        "terrain_dir": args.terrain_dir,
        "terrain_level_min": args.terrain_level_min,
        "terrain_level_max": args.terrain_level_max,
        "num_envs": args.num_envs,
        "duration_s": args.duration,
        "steps": steps,
        "spawn_health": {"initial_termination_counts": _termination_counts(env)},
        "trials": trials,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
