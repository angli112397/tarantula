"""RL PPO actor numpy forward pass.

Stage A structured skid-steer compensation actor:
obs(47) = [projected_gravity_b(3), root_ang_vel_b(3),
           susp_joint_pos(6, LEGS order), susp_joint_vel(6, LEGS order),
           wheel_joint_vel(6, LEGS order), wheel_force(18, LEGS order),
           cmd_vx(1), cmd_wz(1), prev_action(3)]

action(3) = clip(MLP(normalize(obs)), -1, 1)
  action[0] -> bounded effective-track scale correction
  action[1] -> bounded left-drive scale correction
  action[2] -> bounded right-drive scale correction

MLP: Linear(47,128) -> ELU -> Linear(128,128) -> ELU -> Linear(128,3)
obs_normalizer = EmpiricalNormalization (actor_obs_normalization=True).
"""
import numpy as np

_OBS_NORM_EPS = 1e-2  # rsl_rl EmpiricalNormalization default


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.exp(np.minimum(x, 0.0)) - 1.0)


class RLWheelCompensationPolicy:
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
        self.max_abs_wheel_omega = float(npz["max_abs_wheel_omega"][0]) if "max_abs_wheel_omega" in npz else 6.0
        self.track_scale_delta_limit = (
            float(npz["track_scale_delta_limit"][0]) if "track_scale_delta_limit" in npz else 0.30
        )
        self.drive_scale_delta_limit = (
            float(npz["drive_scale_delta_limit"][0]) if "drive_scale_delta_limit" in npz else 0.20
        )
        if self.obs_mean.shape[0] != self.obs_dim:
            raise ValueError(f"obs normalizer has {self.obs_mean.shape[0]} dims, actor expects {self.obs_dim}")
        if self.obs_dim != 47:
            raise ValueError(f"unsupported obs dim {self.obs_dim}; expected current Stage A obs dim 47")
        if self.action_dim != 3:
            raise ValueError(f"unsupported action dim {self.action_dim}; expected current Stage A action dim 3")

    def act(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"obs has {obs.shape[0]} dims, actor expects {self.obs_dim}")
        x = (obs - self.obs_mean) / (self.obs_std + _OBS_NORM_EPS)
        h = _elu(self.w0 @ x + self.b0)
        h = _elu(self.w2 @ h + self.b2)
        out = self.w4 @ h + self.b4
        return np.clip(out, -1.0, 1.0)
