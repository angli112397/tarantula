# Copyright (c) 2026 Tarantula project
# SPDX-License-Identifier: BSD-3-Clause
"""Tarantula Isaac Lab environments."""

import gymnasium as gym

gym.register(
    id="Isaac-Tarantula-Suspension-v0",
    entry_point=f"{__name__}.suspension_env:TarantulaSuspensionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.suspension_env_cfg:TarantulaSuspensionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:TarantulaSuspensionPPORunnerCfg",
    },
)
