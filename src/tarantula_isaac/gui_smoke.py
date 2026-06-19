"""Minimal Isaac Lab GUI smoke for Tarantula spawn and commanded motion.

This is not a training entry point. It starts one lightweight Isaac env, places
the robot on the shared heightmap terrain, sends a fixed cmd_vel-style command,
and prints enough metrics to catch the failures that previously wasted time:

- spawn starts intersecting/sinking into the mesh;
- wheel commands do not move the chassis;
- the chassis mostly spins in place instead of translating for a forward command;
- roll/pitch is immediately unreasonable.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TERRAIN_DIR = REPO_ROOT / "generated" / "terrains" / "rl_curriculum" / "42"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal Isaac Lab GUI smoke for Tarantula.")
parser.add_argument("--terrain-dir", default=str(DEFAULT_TERRAIN_DIR))
parser.add_argument("--terrain-mode", choices=("heightmap", "plane"), default="heightmap")
parser.add_argument("--terrain-level-min", type=int, default=0)
parser.add_argument("--terrain-level-max", type=int, default=0)
parser.add_argument("--settle-seconds", type=float, default=1.0)
parser.add_argument("--drive-seconds", type=float, default=5.0)
parser.add_argument("--cmd-vx", type=float, default=0.12)
parser.add_argument("--cmd-wz", type=float, default=0.0)
parser.add_argument("--drive-scale", type=float, default=None)
parser.add_argument("--yaw-track-scale", type=float, default=None)
parser.add_argument("--push-lin-vel", type=float, default=0.0, help="Max random push velocity for smoke. Default 0 disables training perturbations.")
parser.add_argument("--wall-sleep", type=float, default=0.01, help="Small sleep per sim step so the GUI remains observable.")
parser.add_argument("--min-displacement", type=float, default=0.20)
parser.add_argument("--max-tilt-deg", type=float, default=25.0)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaacsim.core.utils.extensions import enable_extension

from tarantula_isaac.robot import ensure_tarantula_usd
from tarantula_isaac.shared_heightmap_terrain import make_shared_heightmap_terrain_cfg
from tarantula_isaac.suspension_env import TarantulaSuspensionEnv, _quat_roll_pitch
from tarantula_isaac.suspension_env_cfg import TarantulaSuspensionEnvCfg


def _steps(cfg: TarantulaSuspensionEnvCfg, seconds: float) -> int:
    step_dt = float(cfg.sim.dt) * float(cfg.decimation)
    return max(1, int(round(seconds / step_dt)))


def _set_command(env: TarantulaSuspensionEnv, vx: float, wz: float) -> None:
    env._cmd_vx[:] = float(vx)
    env._cmd_wz[:] = float(wz)
    env._update_execution_commands()


def _step(env: TarantulaSuspensionEnv, action: torch.Tensor, count: int, sleep_s: float) -> None:
    for _ in range(count):
        env.step(action)
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def _metrics(env: TarantulaSuspensionEnv) -> dict[str, float]:
    data = env._robot.data
    roll, pitch = _quat_roll_pitch(env._imu.data.quat_w)
    tilt = torch.sqrt(roll * roll + pitch * pitch)
    quat = data.root_quat_w[0]
    w, x, y, z = (float(v.item()) for v in quat)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return {
        "x": float(data.root_pos_w[0, 0].item()),
        "y": float(data.root_pos_w[0, 1].item()),
        "z": float(data.root_pos_w[0, 2].item()),
        "yaw_deg": math.degrees(yaw),
        "roll_deg": math.degrees(float(roll[0].item())),
        "pitch_deg": math.degrees(float(pitch[0].item())),
        "tilt_deg": math.degrees(float(tilt[0].item())),
        "vx_b": float(data.root_lin_vel_b[0, 0].item()),
        "wz_b": float(data.root_ang_vel_b[0, 2].item()),
        "exec_cmd_vx": float(env._exec_cmd_vx[0].item()),
        "exec_cmd_wz": float(env._exec_cmd_wz[0].item()),
    }


def main() -> None:
    enable_extension("isaacsim.asset.importer.urdf")
    ensure_tarantula_usd()

    cfg = TarantulaSuspensionEnvCfg()
    cfg.episode_length_s = max(float(cfg.episode_length_s), float(args.settle_seconds + args.drive_seconds + 2.0))
    if args.drive_scale is not None:
        cfg.drive_scale = float(args.drive_scale)
    if args.yaw_track_scale is not None:
        cfg.yaw_track_scale = float(args.yaw_track_scale)
    push = abs(float(args.push_lin_vel))
    cfg.push_lin_vel_range = (-push, push)
    cfg.push_interval_steps = (10_000_000, 10_000_001)
    cfg.scene.num_envs = 1
    cfg.command_resampling_enabled = False
    cfg.terrain = make_shared_heightmap_terrain_cfg(
        args.terrain_dir,
        min_level=args.terrain_level_min,
        max_level=args.terrain_level_max,
        terrain_type=args.terrain_mode,
        debug_vis=False,
    )

    env = TarantulaSuspensionEnv(cfg=cfg, render_mode=None)
    action = torch.zeros((1, cfg.action_space), device=env.device)

    print(f"[gui_smoke] terrain_dir={args.terrain_dir}", flush=True)
    print(f"[gui_smoke] terrain_mode={args.terrain_mode}", flush=True)
    print(f"[gui_smoke] env_origin={env._terrain.env_origins[0].detach().cpu().numpy().round(3).tolist()}", flush=True)
    print(f"[gui_smoke] cmd_vx={args.cmd_vx:.3f} cmd_wz={args.cmd_wz:.3f}", flush=True)

    _set_command(env, 0.0, 0.0)
    _step(env, action, _steps(cfg, args.settle_seconds), args.wall_sleep)
    start = _metrics(env)
    start_xy = env._robot.data.root_pos_w[0, :2].clone()
    print(f"[gui_smoke] after_settle={start}", flush=True)

    _set_command(env, args.cmd_vx, args.cmd_wz)
    _step(env, action, _steps(cfg, args.drive_seconds), args.wall_sleep)
    end = _metrics(env)
    end_xy = env._robot.data.root_pos_w[0, :2].clone()
    displacement = float(torch.linalg.norm(end_xy - start_xy).item())
    yaw_delta_deg = (end["yaw_deg"] - start["yaw_deg"] + 180.0) % 360.0 - 180.0
    mean_yaw_rate = math.radians(yaw_delta_deg) / max(float(args.drive_seconds), 1.0e-9)
    term = env._termination_terms()
    term_flags = {name: bool(value[0].item()) for name, value in term.items()}

    print(f"[gui_smoke] after_drive={end}", flush=True)
    print(f"[gui_smoke] wheel_target={env._last_wheel_target[0].detach().cpu().numpy().round(4).tolist()}", flush=True)
    print(f"[gui_smoke] displacement_xy={displacement:.3f}m", flush=True)
    print(f"[gui_smoke] yaw_delta={yaw_delta_deg:.2f}deg mean_wz={mean_yaw_rate:.4f}rad/s", flush=True)
    print(f"[gui_smoke] termination_flags={term_flags}", flush=True)

    problems: list[str] = []
    if not math.isfinite(displacement) or displacement < float(args.min_displacement):
        problems.append(f"displacement {displacement:.3f}m < {args.min_displacement:.3f}m")
    if abs(end["tilt_deg"]) > float(args.max_tilt_deg):
        problems.append(f"tilt {end['tilt_deg']:.1f}deg > {args.max_tilt_deg:.1f}deg")
    if any(term_flags.values()):
        problems.append(f"termination flags active: {term_flags}")

    env.close()
    if problems:
        raise RuntimeError("ISAAC_GUI_SMOKE_FAILED: " + "; ".join(problems))
    print("ISAAC_GUI_SMOKE_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
