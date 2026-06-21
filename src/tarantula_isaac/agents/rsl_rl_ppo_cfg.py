# Copyright (c) 2026 Tarantula project
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL PPO runner config for Tarantula active-suspension posture control.

MLP (obs=50, action=6): the policy only commands bounded hip position targets.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class TarantulaSuspensionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 400
    save_interval = 50
    experiment_name = "tarantula_suspension"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    # [512, 256, 128]: the legged_gym/ANYmal-convention actor/critic width for
    # this class of task (proprioceptive obs in the tens of dims, memoryless
    # MLP) -- e.g. Rudin et al. 2021 "Learning to Walk in Minutes" uses this
    # exact shape for ANYmal's ~48D proprioceptive obs. Surveying related
    # work (active-suspension rover, ANYmal blind terrain locomotion, RMA)
    # turned up no PPO actor that benefits from going narrower than this for
    # an obs space our size, so there's no real reason for the smaller
    # [128, 128] we'd shrunk to.
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.35),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
