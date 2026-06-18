"""Export PPO actor weights to a Gazebo-deployable .npz file.

Usage (with isaac_venv active, from repo root):
  python3 src/tarantula_isaac/export_weights_v5.py \\
    --checkpoint logs/rsl_rl/tarantula_suspension/<run>/model_399.pt \\
    --npz-out generated/policies/posture_actor.npz
"""

import argparse
import pathlib
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--npz-out", required=True, help="Path to write raw actor weights as .npz.")
parser.add_argument("--hip-action-target-limit", type=float, default=None)
args = parser.parse_args()

# Isaac Lab imports NOT needed -- torch only
import torch
import numpy as np


def _read_env_yaml_float(checkpoint_path: pathlib.Path, key: str) -> float | None:
    env_yaml = checkpoint_path.parent / "params" / "env.yaml"
    if not env_yaml.exists():
        return None
    prefix = f"{key}:"
    for raw_line in env_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith(prefix):
            continue
        try:
            return float(line.split(":", 1)[1].strip())
        except ValueError:
            return None
    return None

ckpt_path = pathlib.Path(args.checkpoint)
print(f"[export] Loading checkpoint: {ckpt_path}")
ckpt = torch.load(str(ckpt_path), map_location="cpu")

print(f"[export] Top-level keys: {list(ckpt.keys())[:10]}")

# rsl_rl current checkpoint layout:
#   model_state_dict -> actor/critic weights
#   obs_normalizer (or empirical_normalizer) -> mean/var tensors
# alternate rsl_rl checkpoint layouts:
#   model_state_dict -> actor_body.0.weight etc.
#   obs_normalizer._mean / obs_normalizer._var

# Some rsl_rl checkpoints use separate actor_state_dict / critic_state_dict.
model_sd = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))
print(f"[export] Actor state_dict keys: {list(model_sd.keys())}")

# Find actor MLP weights -- key naming varies by rsl_rl version
# Try explicit actor module names, then compact actor layer names.
def find_key(sd, patterns):
    for p in patterns:
        matches = [k for k in sd if p in k]
        if matches:
            return matches
    return []

# common pattern: "actor.model.layers.{0,2,4}.{weight,bias}"
# compact pattern: "actor.{0,2,4}.{weight,bias}"
# Also possible: "actor_body.{0,2,4}.{weight,bias}"
actor_keys = find_key(model_sd, ["actor.model.layers", "actor_body", "actor.0", "actor.2"])
print(f"[export] Candidate actor keys: {actor_keys[:15]}")

npz_data = {}

def _linear_layer_keys(sd: dict, stem: str) -> list[tuple[int, str]]:
    pattern = re.compile(rf"^{re.escape(stem)}\.(\d+)\.weight$")
    layers = []
    for key in sd:
        match = pattern.match(key)
        if not match:
            continue
        idx = int(match.group(1))
        bias_key = f"{stem}.{idx}.bias"
        if bias_key in sd:
            layers.append((idx, f"{stem}.{idx}"))
    return sorted(layers)


linear_layers: list[tuple[int, str]] = []
for stem in ("mlp", "actor.model.layers", "actor_body", "actor"):
    linear_layers = _linear_layer_keys(model_sd, stem)
    if linear_layers:
        print(f"[export] Using actor MLP stem: {stem}, linear layers: {[idx for idx, _ in linear_layers]}")
        break

if not linear_layers:
    print("[ERROR] Could not locate actor MLP weights in checkpoint. Dump all keys:")
    for k in model_sd:
        print(f"  {k}: {model_sd[k].shape}")
    sys.exit(1)

for export_idx, (_, source_prefix) in enumerate(linear_layers):
    npz_data[f"mlp.{export_idx}.weight"] = model_sd[f"{source_prefix}.weight"].numpy()
    npz_data[f"mlp.{export_idx}.bias"] = model_sd[f"{source_prefix}.bias"].numpy()

obs_dim = int(npz_data["mlp.0.weight"].shape[1])
last_layer_idx = len(linear_layers) - 1
action_dim = int(npz_data[f"mlp.{last_layer_idx}.weight"].shape[0])

# Obs normalizer
norm_mean = norm_std = None
for mean_k, var_k in [
    ("obs_normalizer._mean", "obs_normalizer._var"),
    ("obs_normalizer._mean", "obs_normalizer._std"),  # fallback if _std already stored
    ("normalizer._mean",     "normalizer._var"),
    ("obs_normalizer.mean",  "obs_normalizer.var"),
]:
    src = model_sd if mean_k in model_sd else (ckpt if mean_k in ckpt else None)
    if src is None:
        continue
    norm_mean = src[mean_k].numpy().reshape(-1)
    raw_var   = src[var_k].numpy().reshape(-1)
    # _var holds variance; _std holds std; distinguish by key name
    if "_std" in var_k:
        norm_std = np.maximum(raw_var, 1e-2)
    else:
        norm_std = np.sqrt(np.maximum(raw_var, 1e-4))
    print(f"[export] obs_normalizer from: {mean_k}, {var_k}")
    break

if norm_mean is None:
    print("[export] WARNING: obs normalizer not found -- using identity (mean=0, std=1)")
    norm_mean = np.zeros(obs_dim, dtype=np.float32)
    norm_std  = np.ones(obs_dim,  dtype=np.float32)

npz_data["obs_normalizer._mean"] = norm_mean.astype(np.float32)
npz_data["obs_normalizer._std"]  = norm_std.astype(np.float32)
hip_action_target_limit = (
    args.hip_action_target_limit
    if args.hip_action_target_limit is not None
    else _read_env_yaml_float(ckpt_path, "hip_action_target_limit")
)
if hip_action_target_limit is None:
    hip_action_target_limit = 0.30
npz_data["hip_action_target_limit"] = np.asarray([hip_action_target_limit], dtype=np.float32)

# Validate shapes
print(f"[export] MLP shapes:")
for layer_idx in range(len(linear_layers)):
    print(f"  mlp.{layer_idx}.weight: {npz_data[f'mlp.{layer_idx}.weight'].shape}")
    print(f"  mlp.{layer_idx}.bias:   {npz_data[f'mlp.{layer_idx}.bias'].shape}")
print(f"  obs mean:     {npz_data['obs_normalizer._mean'].shape}  (expect [{obs_dim}])")
print(f"  hip target:   {float(npz_data['hip_action_target_limit'][0])} rad")
assert (obs_dim, action_dim) == (50, 6), (
    f"Expected active-suspension 50D/6D actor, got {obs_dim}D/{action_dim}D"
)
assert npz_data["mlp.0.weight"].shape[1] == obs_dim
assert npz_data[f"mlp.{last_layer_idx}.weight"].shape[0] == action_dim
assert npz_data["obs_normalizer._mean"].shape == (obs_dim,), f"Expected ({obs_dim},), got {npz_data['obs_normalizer._mean'].shape}"

npz_path = pathlib.Path(args.npz_out)
npz_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(npz_path, **npz_data)
print(f"[export] Raw npz written to: {npz_path}")
print("[export] DONE - v5 active-suspension posture weights exported.")
