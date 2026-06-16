"""Export PPO actor weights to a Gazebo-deployable .npz file.

Usage (with isaac_venv active, from repo root):
  python3 src/tarantula_isaac/export_weights_v5.py \\
    --checkpoint logs/rsl_rl/tarantula_suspension/<run>/model_399.pt \\
    --npz-out generated/policies/cmd_vel_actor.npz
"""

import argparse
import pathlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--npz-out", required=True, help="Path to write raw actor weights as .npz.")
args = parser.parse_args()

# Isaac Lab imports NOT needed -- torch only
import torch
import numpy as np

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

# Determine actual prefix
npz_data = {}
for prefix_try in [
    ("mlp.0",               "mlp.2",               "mlp.4"),    # flat (rsl_rl v5 actor_state_dict)
    ("actor.model.layers.0","actor.model.layers.2", "actor.model.layers.4"),
    ("actor_body.0",        "actor_body.2",          "actor_body.4"),
    ("actor.0",             "actor.2",               "actor.4"),
]:
    l0, l2, l4 = prefix_try
    if f"{l0}.weight" in model_sd:
        npz_data["mlp.0.weight"] = model_sd[f"{l0}.weight"].numpy()
        npz_data["mlp.0.bias"]   = model_sd[f"{l0}.bias"].numpy()
        npz_data["mlp.2.weight"] = model_sd[f"{l2}.weight"].numpy()
        npz_data["mlp.2.bias"]   = model_sd[f"{l2}.bias"].numpy()
        npz_data["mlp.4.weight"] = model_sd[f"{l4}.weight"].numpy()
        npz_data["mlp.4.bias"]   = model_sd[f"{l4}.bias"].numpy()
        print(f"[export] Using actor key prefix: {l0}")
        break

if not npz_data:
    print("[ERROR] Could not locate actor MLP weights in checkpoint. Dump all keys:")
    for k in model_sd:
        print(f"  {k}: {model_sd[k].shape}")
    sys.exit(1)

obs_dim = int(npz_data["mlp.0.weight"].shape[1])
action_dim = int(npz_data["mlp.4.weight"].shape[0])

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

# Validate shapes
print(f"[export] MLP shapes:")
print(f"  mlp.0.weight: {npz_data['mlp.0.weight'].shape}  (expect [128,41] wheel-only or [128,47] legacy)")
print(f"  mlp.0.bias:   {npz_data['mlp.0.bias'].shape}    (expect [128])")
print(f"  mlp.2.weight: {npz_data['mlp.2.weight'].shape}  (expect [128,128])")
print(f"  mlp.4.weight: {npz_data['mlp.4.weight'].shape}  (expect [6,128] wheel-only or [12,128] legacy)")
print(f"  obs mean:     {npz_data['obs_normalizer._mean'].shape}  (expect [{obs_dim}])")
assert obs_dim in (41, 47), f"Expected obs dim 41 or 47, got {obs_dim}"
assert action_dim in (6, 12), f"Expected action dim 6 or 12, got {action_dim}"
assert npz_data["mlp.0.weight"].shape == (128, obs_dim), f"Expected (128,{obs_dim}), got {npz_data['mlp.0.weight'].shape}"
assert npz_data["mlp.4.weight"].shape == (action_dim, 128), f"Expected ({action_dim},128), got {npz_data['mlp.4.weight'].shape}"
assert npz_data["obs_normalizer._mean"].shape == (obs_dim,), f"Expected ({obs_dim},), got {npz_data['obs_normalizer._mean'].shape}"

npz_path = pathlib.Path(args.npz_out)
npz_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(npz_path, **npz_data)
print(f"[export] Raw npz written to: {npz_path}")
print("[export] DONE — v5 weights exported.")
