"""RL PPO actor numpy forward pass.

Structured skid-steer compensation actor:
obs(47) = [projected_gravity_b(3), root_ang_vel_b(3),
           susp_joint_pos(6, LEGS order), susp_joint_vel(6, LEGS order),
           wheel_joint_vel(6, LEGS order), wheel_force(18, LEGS order),
           cmd_vx(1), cmd_wz(1), prev_action(3)]

action(3) = clip(MLP(normalize(obs)), -1, 1)
  action[0] -> bounded effective-track scale correction
  action[1] -> bounded left-drive scale correction
  action[2] -> bounded right-drive scale correction

MLP is exported as ordered ``mlp.N.weight/bias`` arrays. The default Stage B
network is Linear(53,512) -> ELU -> Linear(512,256) -> ELU ->
Linear(256,128) -> ELU -> Linear(128,9).
obs_normalizer = EmpiricalNormalization (actor_obs_normalization=True).

Stage B is the current baseline: the first 44 observation values are
proprioception/command terms and the final 9 values are prev_action, for a 53D
actor. Its 9D action keeps the first three wheel residual dimensions and adds
action[3:9] as direct hip position targets, LEGS order. Legacy Stage A 47D/3D
actors remain loadable for ablation and regression comparison.
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
        self.max_abs_wheel_omega = float(npz["max_abs_wheel_omega"][0]) if "max_abs_wheel_omega" in npz else 6.0
        self.track_scale_delta_limit = (
            float(npz["track_scale_delta_limit"][0]) if "track_scale_delta_limit" in npz else 0.30
        )
        self.drive_scale_delta_limit = (
            float(npz["drive_scale_delta_limit"][0]) if "drive_scale_delta_limit" in npz else 0.20
        )
        self.hip_action_target_limit = (
            float(npz["hip_action_target_limit"][0]) if "hip_action_target_limit" in npz else 0.30
        )
        if self.obs_mean.shape[0] != self.obs_dim:
            raise ValueError(f"obs normalizer has {self.obs_mean.shape[0]} dims, actor expects {self.obs_dim}")
        if (self.obs_dim, self.action_dim) not in {(47, 3), (53, 9)}:
            raise ValueError(
                f"unsupported actor shape {self.obs_dim}D/{self.action_dim}D; "
                "expected Stage A 47D/3D or Stage B 53D/9D"
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
