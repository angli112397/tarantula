# Copyright (c) 2026 Tarantula project
# SPDX-License-Identifier: BSD-3-Clause
"""Procedural training terrain for the tarantula 6-wheel rover (Isaac Lab).

Slope/noise bounds are derived from already-validated Gazebo scenarios rather than
IsaacLab's quadruped defaults (``isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG``),
which use step heights up to 0.23 m and slopes up to ~23 deg -- both far beyond what
this rover's suspension envelope has been shown to handle (see
docs/01-control-architecture.md "已知限制" and docs/03-isaac-lab-setup.md).

Design constraints (tarantula geometry: wheel_radius=0.12 m, track ~0.8 m,
arm_length=0.22 m, see tarantula_chassis.xacro):

* v4 adds ``hf_discrete_obstacles`` (height 0.04-0.10 m, width 0.10-0.20 m) to
  close the train-deploy terrain gap: Gazebo rough_terrain.world has cylinder
  bumps (radius 0.06/0.08 m) and half-steps (0.05/0.06 m) that were absent from
  the v3 training distribution, causing the policy to never encounter bumps during
  training. Obstacle height is capped at ~0.83x wheel_radius (0.12 m) to stay
  below the geometric rollover threshold for a single-wheel contact.
* ``slope_range`` is capped at 0.15 rad (~8.6 deg), matching the validated G2
  static-tilt scenario (tilt_test.world, 8 deg, passive 8.16 deg / active 0.09 deg,
  no rollover) and the G3 rough_terrain ramp pitch (~8.6 deg).
* ``platform_width`` (flat spawn area at the center of each sub-terrain tile) is
  set well above the rover's footprint diagonal (~1.6 m) so episodes start on
  flat ground.
* ``random_rough`` noise amplitude is capped at 0.03 m (a quarter of the wheel
  radius) -- enough surface roughness to be useful without reproducing the
  half-step (0.05-0.06 m) geometry that already stresses roll dynamics in G3.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

TARANTULA_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=4.0,
    num_rows=5,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.0, 0.0), noise_step=0.005, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20, noise_range=(0.0, 0.03), noise_step=0.005, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20, slope_range=(0.0, 0.15), platform_width=2.5, border_width=0.5
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.15), platform_width=2.5, border_width=0.5
        ),
        "hf_wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.10, amplitude_range=(0.0, 0.1), num_waves=1, border_width=0.25
        ),
        # v4: discrete obstacles to match Gazebo rough_terrain.world bumps/half-steps
        # (radius 0.06-0.08 m cylinder / height 0.05-0.06 m box). Proportion 0.20
        # ensures the policy sees bumps frequently during curriculum training.
        "hf_discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            platform_width=2.0,
            border_width=0.25,
            obstacle_height_mode="choice",
            obstacle_height_range=(0.04, 0.10),
            obstacle_width_range=(0.10, 0.20),
            num_obstacles=5,
        ),
    },
)
"""Curriculum terrain generator config for tarantula RL training (M5/M7).

Difficulty 0 -> flat ground (matches M4 baseline). Difficulty 1 -> ~8.6 deg
slopes / 3 cm surface noise, both within the rover's already-validated envelope.
"""
