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
parser.add_argument("--pursuit", action="store_true",
                     help="Drive pure-pursuit checkpoint chasing (cmd_vx as cruise speed) instead of a fixed cmd_vel.")
parser.add_argument("--pursuit-checkpoints", type=int, default=None,
                     help="Override CommandsCfg.pursuit_checkpoint_count for this smoke run.")
parser.add_argument("--drive-scale", type=float, default=None)
parser.add_argument("--yaw-track-scale", type=float, default=None)
parser.add_argument("--push-lin-vel", type=float, default=0.0, help="Max random push velocity for smoke. Default 0 disables training perturbations.")
parser.add_argument("--wall-sleep", type=float, default=0.01, help="Small sleep per sim step so the GUI remains observable.")
parser.add_argument("--min-displacement", type=float, default=0.20)
parser.add_argument("--max-tilt-deg", type=float, default=25.0)
parser.add_argument("--policy-weights-npz", default=None,
                     help="If given, drive hip targets with this trained RLPosturePolicy actor every step "
                          "instead of the fixed zero action used otherwise, and report per-observation-slice "
                          "health stats (NaN/Inf, range, a few physically-implausible-value checks) at the end. "
                          "Reuses tarantula_control.rl_policy.RLPosturePolicy -- the exact deployable inference "
                          "path Gazebo's posture_policy_node.py uses, not a separate Isaac-native forward pass.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = False
if args.pursuit and args.drive_seconds == parser.get_default("drive_seconds"):
    # A single checkpoint chase at smoke-test cruise speed routinely takes
    # 30-60s+ depending on sampled distance -- the 5s fixed-cmd_vel default
    # would barely move before the test ends.
    args.drive_seconds = 60.0

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaacsim.core.utils.extensions import enable_extension

from tarantula_control.control_interfaces import MAX_ABS_WHEEL_OMEGA
from tarantula_control.motion_control import POSTURE_OBSERVATION_LAYOUT
from tarantula_control.rl_policy import RLPosturePolicy
from tarantula_control.suspension_core import HIP_TARGET_LIMIT
from tarantula_isaac.robot import ensure_tarantula_usd
from tarantula_isaac.shared_heightmap_terrain import make_shared_heightmap_terrain_cfg
from tarantula_isaac.suspension_env import TarantulaSuspensionEnv, _quat_roll_pitch
from tarantula_isaac.suspension_env_cfg import TarantulaSuspensionEnvCfg


class ObsHealthTracker:
    """Accumulates per-named-slice min/max/mean/NaN-Inf counts across a long
    run, using POSTURE_OBSERVATION_LAYOUT (motion_control.py's single source
    of truth for the deployable 56D contract) for slice boundaries instead
    of hand-duplicated indices -- if that layout ever changes, this tracks
    it automatically rather than silently checking the wrong columns."""

    def __init__(self):
        self.bounds: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, dim, _desc in POSTURE_OBSERVATION_LAYOUT:
            self.bounds[name] = (offset, offset + dim)
            offset += dim
        self.mins = {name: None for name in self.bounds}
        self.maxs = {name: None for name in self.bounds}
        self.sums = {name: 0.0 for name in self.bounds}
        self.count = 0
        self.nan_inf_steps = 0
        self.gravity_norm_violations = 0
        self.hip_limit_violations = 0
        self.wheel_omega_violations = 0
        self.wheel_force_saturated_steps = 0
        self.contact_uptime_range_violations = 0

    def update(self, obs_row: np.ndarray) -> None:
        if not np.all(np.isfinite(obs_row)):
            self.nan_inf_steps += 1
            return  # don't let a NaN/Inf step corrupt min/max/mean stats
        self.count += 1
        for name, (lo, hi) in self.bounds.items():
            seg = obs_row[lo:hi]
            seg_min, seg_max = float(seg.min()), float(seg.max())
            self.mins[name] = seg_min if self.mins[name] is None else min(self.mins[name], seg_min)
            self.maxs[name] = seg_max if self.maxs[name] is None else max(self.maxs[name], seg_max)
            self.sums[name] += float(seg.sum())

        grav_lo, grav_hi = self.bounds["projected_gravity_b"]
        if abs(float(np.linalg.norm(obs_row[grav_lo:grav_hi])) - 1.0) > 0.05:
            self.gravity_norm_violations += 1
        hip_lo, hip_hi = self.bounds["susp_joint_pos"]
        if np.any(np.abs(obs_row[hip_lo:hip_hi]) > HIP_TARGET_LIMIT + 0.05):
            self.hip_limit_violations += 1
        wheel_lo, wheel_hi = self.bounds["wheel_joint_vel"]
        if np.any(np.abs(obs_row[wheel_lo:wheel_hi]) > MAX_ABS_WHEEL_OMEGA + 0.5):
            self.wheel_omega_violations += 1
        force_lo, force_hi = self.bounds["wheel_force"]
        if np.any(np.abs(obs_row[force_lo:force_hi]) >= 2.999):
            self.wheel_force_saturated_steps += 1
        uptime_lo, uptime_hi = self.bounds["contact_uptime"]
        seg = obs_row[uptime_lo:uptime_hi]
        if np.any(seg < -1.0e-3) or np.any(seg > 1.0 + 1.0e-3):
            self.contact_uptime_range_violations += 1

    def report_lines(self) -> list[str]:
        lines = [f"[gui_smoke] observation health: {self.count} steps checked, "
                 f"{self.nan_inf_steps} NaN/Inf steps (excluded from min/max/mean below)"]
        for name, (lo, hi) in self.bounds.items():
            dim = hi - lo
            mean = self.sums[name] / max(self.count * dim, 1)
            lines.append(
                f"[gui_smoke]   {name:18s} dim={dim:2d} min={self.mins[name]:+9.4f} "
                f"max={self.maxs[name]:+9.4f} mean={mean:+9.4f}"
            )
        lines.append(f"[gui_smoke] projected_gravity_b |norm-1|>0.05: {self.gravity_norm_violations} steps "
                     "(should be a unit vector every step)")
        lines.append(f"[gui_smoke] susp_joint_pos beyond URDF limit (+-{HIP_TARGET_LIMIT}+0.05 rad): "
                     f"{self.hip_limit_violations} steps")
        lines.append(f"[gui_smoke] wheel_joint_vel beyond MAX_ABS_WHEEL_OMEGA+0.5 ({MAX_ABS_WHEEL_OMEGA}+0.5 rad/s): "
                     f"{self.wheel_omega_violations} steps")
        lines.append(f"[gui_smoke] wheel_force saturated at the +-3.0 normalize-clamp: "
                     f"{self.wheel_force_saturated_steps} steps")
        lines.append(f"[gui_smoke] contact_uptime outside [0,1]: {self.contact_uptime_range_violations} steps")
        return lines


def _steps(cfg: TarantulaSuspensionEnvCfg, seconds: float) -> int:
    step_dt = float(cfg.sim.dt) * float(cfg.decimation)
    return max(1, int(round(seconds / step_dt)))


def _set_command(env: TarantulaSuspensionEnv, vx: float, wz: float) -> None:
    env._cmd_vx[:] = float(vx)
    env._cmd_wz[:] = float(wz)
    env._update_execution_commands()


def _policy_action(policy: RLPosturePolicy, obs: torch.Tensor) -> torch.Tensor:
    """One inference call through the exact deployable path (numpy forward
    pass, see rl_policy.py) -- obs is (1,56) on whatever device env uses;
    policy.act() is numpy/CPU-only, so round-trip through .cpu().numpy()."""
    obs_np = obs[0].detach().cpu().numpy().astype(np.float32)
    action_np = policy.act(obs_np)
    return torch.as_tensor(action_np, dtype=torch.float32, device=obs.device).unsqueeze(0)


def _drive_pursuit(env: TarantulaSuspensionEnv, action: torch.Tensor, cmd_vx: float, count: int,
                    sleep_s: float, step_dt: float, *,
                    policy: RLPosturePolicy | None = None, tracker: ObsHealthTracker | None = None) -> None:
    """Start a pure-pursuit checkpoint chase for env 0 and step through it,
    logging on every checkpoint/command-mode transition so the GUI run is
    legible without printing every step. _update_pursuit_commands (called
    each step via _get_observations -> _advance_command_curriculum) recomputes
    cmd_wz from heading error on its own; this just starts the chase, holds a
    fixed cruise cmd_vx, and narrates.

    With policy given, ``action`` is only the seed for step 0 -- every
    subsequent action is recomputed from the live observation (the standard
    obs -> policy -> env.step(action) -> obs loop), and a checkpoint
    sequence exhausting mid-run (falls back to command_mode 0, see
    _update_pursuit_commands) is detected and restarted so a long run stays
    in pursuit mode throughout rather than drifting into an unrelated
    primitive command for the remainder."""
    env_ids = torch.tensor([0], device=env.device)
    env._start_pursuit_command(env_ids)
    env._cmd_vx[env_ids] = float(cmd_vx)
    env._update_execution_commands()
    print(f"[gui_smoke] pursuit started: checkpoint_count={int(env._pursuit_checkpoints_left[0].item())} "
          f"cruise_vx={cmd_vx:.3f} waypoint_0={env._pursuit_waypoint[0].detach().cpu().numpy().round(3).tolist()}",
          flush=True)

    obs = env._get_observations()["policy"] if policy is not None else None
    last_checkpoints_left = int(env._pursuit_checkpoints_left[0].item())
    last_mode = int(env._command_mode[0].item())
    last_heartbeat = -1
    restarts = 0
    resets = 0
    for step_i in range(count):
        step_action = _policy_action(policy, obs) if policy is not None else action
        obs_dict, _rew, terminated, truncated, _info = env.step(step_action)
        if policy is not None:
            obs = obs_dict["policy"]
            if tracker is not None:
                tracker.update(obs[0].detach().cpu().numpy())
        if bool(terminated[0].item()) or bool(truncated[0].item()):
            resets += 1
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        t_s = step_i * step_dt
        checkpoints_left = int(env._pursuit_checkpoints_left[0].item())
        mode = int(env._command_mode[0].item())
        pos = env._robot.data.root_pos_w[0, :2].detach().cpu().numpy().round(3).tolist()
        if mode != 2:
            # Either a checkpoint sequence ran out (see _update_pursuit_commands's
            # command_mode reset to 0) or env 0 was just reset (also lands in
            # primitive mode) -- either way, keep this a pursuit-mode demo.
            env._start_pursuit_command(env_ids)
            env._cmd_vx[env_ids] = float(cmd_vx)
            env._update_execution_commands()
            restarts += 1
            print(f"[gui_smoke] t={t_s:5.1f}s pos={pos} pursuit restarted (was command_mode={mode}, "
                  f"restart #{restarts})", flush=True)
            mode = 2
        if checkpoints_left != last_checkpoints_left or mode != last_mode:
            print(f"[gui_smoke] t={t_s:5.1f}s pos={pos} command_mode={mode} "
                  f"pursuit_checkpoints_left={checkpoints_left} "
                  f"waypoint={env._pursuit_waypoint[0].detach().cpu().numpy().round(3).tolist()}", flush=True)
            last_checkpoints_left, last_mode = checkpoints_left, mode
        elif int(t_s // 5.0) != last_heartbeat:
            last_heartbeat = int(t_s // 5.0)
            to_target = env._pursuit_waypoint[0].detach().cpu().numpy() - pos
            distance = float((to_target[0] ** 2 + to_target[1] ** 2) ** 0.5)
            print(f"[gui_smoke] t={t_s:5.1f}s pos={pos} distance_to_waypoint={distance:.2f}m "
                  f"cmd_wz={float(env._exec_cmd_wz[0].item()):.3f}", flush=True)
    if resets > 0:
        print(f"[gui_smoke] env 0 reset (termination/truncation) {resets} time(s) during this run", flush=True)


def _step(env: TarantulaSuspensionEnv, action: torch.Tensor, count: int, sleep_s: float, *,
          policy: RLPosturePolicy | None = None, tracker: ObsHealthTracker | None = None) -> None:
    obs = env._get_observations()["policy"] if policy is not None else None
    for _ in range(count):
        step_action = _policy_action(policy, obs) if policy is not None else action
        obs_dict, _rew, _terminated, _truncated, _info = env.step(step_action)
        if policy is not None:
            obs = obs_dict["policy"]
            if tracker is not None:
                tracker.update(obs[0].detach().cpu().numpy())
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
    cfg.domain_rand.push_lin_vel_range = (-push, push)
    # Disabled sentinel only when push is actually off — otherwise the
    # interval never fires within a short smoke run and --push-lin-vel is a
    # no-op despite the configured magnitude.
    cfg.domain_rand.push_interval_steps = (150, 300) if push > 0.0 else (10_000_000, 10_000_001)
    cfg.scene.num_envs = 1
    cfg.commands.resampling_enabled = False
    if args.pursuit_checkpoints is not None:
        cfg.commands.pursuit_checkpoint_count = int(args.pursuit_checkpoints)
    cfg.terrain = make_shared_heightmap_terrain_cfg(
        args.terrain_dir,
        min_level=args.terrain_level_min,
        max_level=args.terrain_level_max,
        terrain_type=args.terrain_mode,
        debug_vis=False,
    )

    env = TarantulaSuspensionEnv(cfg=cfg, render_mode=None)
    action = torch.zeros((1, cfg.action_space), device=env.device)

    policy: RLPosturePolicy | None = None
    tracker: ObsHealthTracker | None = None
    if args.policy_weights_npz:
        policy = RLPosturePolicy(args.policy_weights_npz)
        tracker = ObsHealthTracker()
        print(f"[gui_smoke] policy loaded: {args.policy_weights_npz} "
              f"(obs_dim={policy.obs_dim}, action_dim={policy.action_dim}, "
              f"hip_limit={policy.hip_action_target_limit})", flush=True)

    print(f"[gui_smoke] terrain_dir={args.terrain_dir}", flush=True)
    print(f"[gui_smoke] terrain_mode={args.terrain_mode}", flush=True)
    print(f"[gui_smoke] env_origin={env._terrain.env_origins[0].detach().cpu().numpy().round(3).tolist()}", flush=True)
    print(f"[gui_smoke] cmd_vx={args.cmd_vx:.3f} cmd_wz={args.cmd_wz:.3f}", flush=True)

    # Settle always zero-action regardless of --policy-weights-npz: a
    # consistent, simple neutral resting pose before anything (policy or
    # fixed command) takes over, not something worth varying per run.
    _set_command(env, 0.0, 0.0)
    _step(env, action, _steps(cfg, args.settle_seconds), args.wall_sleep)
    start = _metrics(env)
    start_xy = env._robot.data.root_pos_w[0, :2].clone()
    print(f"[gui_smoke] after_settle={start}", flush=True)

    if args.pursuit:
        step_dt = float(cfg.sim.dt) * float(cfg.decimation)
        _drive_pursuit(env, action, args.cmd_vx, _steps(cfg, args.drive_seconds), args.wall_sleep, step_dt,
                        policy=policy, tracker=tracker)
    else:
        _set_command(env, args.cmd_vx, args.cmd_wz)
        _step(env, action, _steps(cfg, args.drive_seconds), args.wall_sleep, policy=policy, tracker=tracker)
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

    if tracker is not None:
        for line in tracker.report_lines():
            print(line, flush=True)
        if tracker.nan_inf_steps > 0:
            print(f"[gui_smoke] WARNING: {tracker.nan_inf_steps} steps had NaN/Inf in the observation", flush=True)

    problems: list[str] = []
    if not math.isfinite(displacement) or displacement < float(args.min_displacement):
        problems.append(f"displacement {displacement:.3f}m < {args.min_displacement:.3f}m")
    if abs(end["tilt_deg"]) > float(args.max_tilt_deg):
        problems.append(f"tilt {end['tilt_deg']:.1f}deg > {args.max_tilt_deg:.1f}deg")
    if any(term_flags.values()):
        problems.append(f"termination flags active: {term_flags}")
    if tracker is not None and tracker.nan_inf_steps > 0:
        problems.append(f"{tracker.nan_inf_steps} NaN/Inf observation steps")

    env.close()
    if problems:
        raise RuntimeError("ISAAC_GUI_SMOKE_FAILED: " + "; ".join(problems))
    print("ISAAC_GUI_SMOKE_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
