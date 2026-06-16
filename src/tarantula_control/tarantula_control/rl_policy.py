"""M7 v5 RL suspension+drive policy -- PPO actor numpy forward pass.

obs(47) = [projected_gravity_b(3), root_ang_vel_b(3), root_lin_vel_b(3),
           susp_joint_pos(6, LEGS order), susp_joint_vel(6, LEGS order),
           wheel_joint_vel(6, LEGS order), wheel_in_contact(6, LEGS order),
           move_cmd(1), heading_rate_cmd(1), prev_action(12)]

action(12) = clip(MLP(normalize(obs)), -1, 1)
  action[0:6]  -> susp joint angle targets * ACTION_SCALE_SUSP (rad)
  action[6:12] -> wheel velocity targets  * ACTION_SCALE_WHEEL_OMEGA (rad/s)

MLP: Linear(47,128) -> ELU -> Linear(128,128) -> ELU -> Linear(128,12)
obs_normalizer = EmpiricalNormalization (actor_obs_normalization=True).
"""
import base64
import io

import numpy as np

from .rl_policy_weights import ACTOR_WEIGHTS_NPZ_B64

# Must match suspension_env_cfg.py
ACTION_SCALE_SUSP = 0.5         # rad, direct susp joint angle
ACTION_SCALE_WHEEL_OMEGA = 3.0  # rad/s, per-wheel velocity

_OBS_NORM_EPS = 1e-2  # rsl_rl EmpiricalNormalization default


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.exp(np.minimum(x, 0.0)) - 1.0)


class RLSuspensionPolicy:
    """Stateless MLP: obs(47,) -> action(12,), clipped to [-1, 1]."""

    def __init__(self):
        npz = np.load(io.BytesIO(base64.b64decode(ACTOR_WEIGHTS_NPZ_B64)))
        self.w0, self.b0 = npz["mlp.0.weight"], npz["mlp.0.bias"]
        self.w2, self.b2 = npz["mlp.2.weight"], npz["mlp.2.bias"]
        self.w4, self.b4 = npz["mlp.4.weight"], npz["mlp.4.bias"]
        self.obs_mean = npz["obs_normalizer._mean"].reshape(-1)
        self.obs_std = npz["obs_normalizer._std"].reshape(-1)

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.obs_mean) / (self.obs_std + _OBS_NORM_EPS)
        h = _elu(self.w0 @ x + self.b0)
        h = _elu(self.w2 @ h + self.b2)
        out = self.w4 @ h + self.b4
        return np.clip(out, -1.0, 1.0)
