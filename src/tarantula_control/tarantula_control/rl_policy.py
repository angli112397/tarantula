"""RL PPO actor numpy forward pass.

Stage A wheel-only actor:
obs(41) = [projected_gravity_b(3), root_ang_vel_b(3), root_lin_vel_b(3),
           susp_joint_pos(6, LEGS order), susp_joint_vel(6, LEGS order),
           wheel_joint_vel(6, LEGS order), wheel_load(6, LEGS order),
           cmd_vx(1), cmd_wz(1), prev_action(6)]

action(6) = clip(MLP(normalize(obs)), -1, 1)
  action[0:6] -> wheel velocity targets * ACTION_SCALE_WHEEL_OMEGA (rad/s)

Legacy suspension+wheel actor:
obs(47) = same prefix + prev_action(12)

action(12) = clip(MLP(normalize(obs)), -1, 1)
  action[0:6]  -> susp joint angle targets * ACTION_SCALE_SUSP (rad)
  action[6:12] -> wheel velocity targets  * ACTION_SCALE_WHEEL_OMEGA (rad/s)

MLP: Linear(obs_dim,128) -> ELU -> Linear(128,128) -> ELU -> Linear(128,action_dim)
obs_normalizer = EmpiricalNormalization (actor_obs_normalization=True).
"""
import numpy as np

# Must match suspension_env_cfg.py
ACTION_SCALE_SUSP = 0.5         # rad, direct susp joint angle
ACTION_SCALE_WHEEL_OMEGA = 3.0  # rad/s, per-wheel velocity

_OBS_NORM_EPS = 1e-2  # rsl_rl EmpiricalNormalization default


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.exp(np.minimum(x, 0.0)) - 1.0)


class RLSuspensionPolicy:
    """Stateless MLP actor, clipped to [-1, 1]."""

    def __init__(self, weights_npz_path: str = ""):
        if not weights_npz_path:
            raise ValueError("policy_weights_npz is required for the cmd_vel RL policy")
        npz = np.load(weights_npz_path)
        self.w0, self.b0 = npz["mlp.0.weight"], npz["mlp.0.bias"]
        self.w2, self.b2 = npz["mlp.2.weight"], npz["mlp.2.bias"]
        self.w4, self.b4 = npz["mlp.4.weight"], npz["mlp.4.bias"]
        self.obs_mean = npz["obs_normalizer._mean"].reshape(-1)
        self.obs_std = npz["obs_normalizer._std"].reshape(-1)
        self.obs_dim = int(self.w0.shape[1])
        self.action_dim = int(self.w4.shape[0])
        if self.obs_mean.shape[0] != self.obs_dim:
            raise ValueError(f"obs normalizer has {self.obs_mean.shape[0]} dims, actor expects {self.obs_dim}")
        if self.action_dim not in (6, 12):
            raise ValueError(f"unsupported action dim {self.action_dim}; expected 6 or 12")

    def act(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"obs has {obs.shape[0]} dims, actor expects {self.obs_dim}")
        x = (obs - self.obs_mean) / (self.obs_std + _OBS_NORM_EPS)
        h = _elu(self.w0 @ x + self.b0)
        h = _elu(self.w2 @ h + self.b2)
        out = self.w4 @ h + self.b4
        return np.clip(out, -1.0, 1.0)
