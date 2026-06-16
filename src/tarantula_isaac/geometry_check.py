# Copyright (c) 2026 Tarantula project
# SPDX-License-Identifier: BSD-3-Clause
"""Headless geometry/drive regression test for Isaac-Tarantula-Suspension-v0.

Two checks, each against an expectation independent of "did it run without
error":

1. Settle: zero cmd_vel, run ~2s, then check the chassis settles within
   SPAWN_GROUND_CLEARANCE..+0.10m of SPAWN_Z_OFFSET (the URDF-derived "wheels
   on flat ground" height) and stays level (projected_gravity_b ~= [0,0,-1]).
   Catches spawn-height/leg-geometry mismatches (wheels floating above or
   sinking through the terrain).
2. Drive: cmd_vel=1.0 m/s, run ~3s, then check xy displacement tracks
   cmd_vel*t and wheel joint_vel tracks the commanded omega. Catches
   "wheels spin but the chassis doesn't move" traction failures.

Prints GEOMETRY_CHECK_OK and exits 0 on success; raises AssertionError
(exit 1) with a descriptive message on failure.
"""

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest of imports must come after SimulationApp launch.
import gymnasium as gym
import torch

import tarantula_isaac  # noqa: F401 -- registers Isaac-Tarantula-Suspension-v0
from tarantula_isaac.robot import SPAWN_GROUND_CLEARANCE, SPAWN_Z_OFFSET, URDF_PATH, ensure_tarantula_usd
from tarantula_isaac.suspension_env_cfg import TarantulaSuspensionEnvCfg


def main() -> None:
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.urdf")
    ensure_tarantula_usd(URDF_PATH)

    env_cfg = TarantulaSuspensionEnvCfg()
    env_cfg.scene.num_envs = 2
    env = gym.make("Isaac-Tarantula-Suspension-v0", cfg=env_cfg)
    env.reset()

    base_env = env.unwrapped
    zero_action = torch.zeros(base_env.num_envs, 3, device=base_env.device)
    steps_per_sec = round(1.0 / (env_cfg.sim.dt * env_cfg.decimation))

    # --- Check 1: settle on flat ground with no drive command ---
    base_env._cmd_vel[:] = 0.0
    for _ in range(2 * steps_per_sec):
        env.step(zero_action)

    data = base_env._robot.data
    root_h = data.root_pos_w[:, 2]
    settle_drop = SPAWN_Z_OFFSET - root_h
    grav_z = data.projected_gravity_b[:, 2]

    print(f"[settle] SPAWN_Z_OFFSET={SPAWN_Z_OFFSET:.4f}")
    print(f"[settle] root height z: {root_h.tolist()}")
    print(f"[settle] settle_drop (SPAWN_Z_OFFSET - root_h): {settle_drop.tolist()}")
    print(f"[settle] projected_gravity_b.z: {grav_z.tolist()}")

    assert (settle_drop > -0.02).all(), f"chassis floated up unexpectedly: settle_drop={settle_drop.tolist()}"
    assert (settle_drop < 0.10).all(), (
        f"chassis sank {settle_drop.tolist()} m below the URDF-derived spawn height "
        f"(SPAWN_Z_OFFSET={SPAWN_Z_OFFSET:.4f}, clearance={SPAWN_GROUND_CLEARANCE}) -- "
        "wheels may not be resting on the terrain (leg geometry / spawn height mismatch)"
    )
    assert (grav_z < -0.95).all(), f"chassis not resting level: projected_gravity_b.z={grav_z.tolist()}"

    # --- Check 2: drive at a known cmd_vel and verify displacement tracks it ---
    init_pos = data.root_pos_w[:, :2].clone()
    base_env._cmd_vel[:] = 1.0
    target_omega = 1.0 / env_cfg.wheel_radius

    drive_seconds = 3
    for _ in range(drive_seconds * steps_per_sec):
        env.step(zero_action)

    data = base_env._robot.data
    disp = (data.root_pos_w[:, :2] - init_pos).norm(dim=-1)
    expected_disp = 1.0 * drive_seconds
    wheel_vel = data.joint_vel[:, base_env._wheel_joint_ids].mean(dim=-1)

    print(f"[drive] displacement xy: {disp.tolist()} (expected ~{expected_disp})")
    print(f"[drive] wheel_vel mean: {wheel_vel.tolist()} (target {target_omega:.3f})")

    assert (disp > 0.5 * expected_disp).all(), (
        f"displacement {disp.tolist()} m is far below the expected {expected_disp} m "
        "at cmd_vel=1.0 m/s for 3s -- wheels may be spinning without traction"
    )
    assert ((wheel_vel - target_omega).abs() < 0.5 * target_omega).all(), (
        f"wheel_vel {wheel_vel.tolist()} rad/s far from commanded target {target_omega:.3f} rad/s"
    )

    env.close()
    print("GEOMETRY_CHECK_OK")


if __name__ == "__main__":
    main()
    simulation_app.close()
