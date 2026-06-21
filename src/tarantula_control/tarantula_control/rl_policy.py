"""Gazebo-deployable active-suspension PPO actor forward pass.

The deployable actor is fixed at 56D observation / 6D action:

- obs[0:50]: IMU, hip state, wheel velocity, wheel F/T, per-leg contact
  uptime (~1s EMA), and limited cmd_vx/cmd_wz
- obs[50:56]: previous six hip actions
- action[0:6]: direct bounded hip position targets in LEGS order
"""

from __future__ import annotations

import numpy as np

_OBS_NORM_EPS = 1e-2  # rsl_rl EmpiricalNormalization default


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.exp(np.minimum(x, 0.0)) - 1.0)


class RLPosturePolicy:
    """Stateless MLP actor for six-hip active-suspension control."""

    def __init__(self, weights_npz_path: str = ""):
        if not weights_npz_path:
            raise ValueError("policy_weights_npz is required for the active-suspension policy")
        npz = np.load(weights_npz_path)
        self.layers = []
        layer_idx = 0
        while f"mlp.{layer_idx}.weight" in npz:
            self.layers.append((npz[f"mlp.{layer_idx}.weight"], npz[f"mlp.{layer_idx}.bias"]))
            layer_idx += 1
        if len(self.layers) < 2:
            raise ValueError("policy MLP must contain at least input and output linear layers")
        self.obs_mean = npz["obs_normalizer._mean"].reshape(-1)
        self.obs_std = npz["obs_normalizer._std"].reshape(-1)
        self.obs_dim = int(self.layers[0][0].shape[1])
        self.action_dim = int(self.layers[-1][0].shape[0])
        # Fallback only fires if a checkpoint was exported without embedding
        # this value -- 0.25 matches TarantulaSuspensionEnvCfg's training
        # default as of the checkpoints that predate this field (current
        # default is 0.35; any checkpoint trained since this field existed
        # always embeds its own real value, so this never overrides those).
        self.hip_action_target_limit = (
            float(npz["hip_action_target_limit"][0]) if "hip_action_target_limit" in npz else 0.25
        )
        if self.obs_mean.shape[0] != self.obs_dim:
            raise ValueError(f"obs normalizer has {self.obs_mean.shape[0]} dims, actor expects {self.obs_dim}")
        if (self.obs_dim, self.action_dim) != (56, 6):
            raise ValueError(
                f"unsupported actor shape {self.obs_dim}D/{self.action_dim}D; "
                "expected active-suspension 56D/6D"
            )

    def act(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"obs has {obs.shape[0]} dims, actor expects {self.obs_dim}")
        x = (obs - self.obs_mean) / (self.obs_std + _OBS_NORM_EPS)
        h = x
        for weight, bias in self.layers[:-1]:
            h = _elu(weight @ h + bias)
        weight, bias = self.layers[-1]
        out = weight @ h + bias
        return np.clip(out, -1.0, 1.0)
