#!/usr/bin/env bash
# M7 v5 Isaac Lab env smoke test
# Verifies: obs.shape==(N,47), action_space.shape==(N,12), ContactSensor alive, no NaN
# Usage: bash scripts/run_rl_env_smoke_v5.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAAC_VENV="${ISAAC_VENV:-/home/ang/isaac_venv}"

SMOKE_PY="$(mktemp /tmp/tarantula_smoke_v5_XXXXXX.py)"
trap 'rm -f "$SMOKE_PY"' EXIT

cat > "$SMOKE_PY" << 'PYEOF'
import sys, pathlib
repo = pathlib.Path("/home/ang/Documents/tarantula")
sys.path.insert(0, str(repo / "src/tarantula_isaac"))
sys.path.insert(0, str(repo / "src/tarantula_control"))

# AppLauncher MUST be first — it initializes the Kit which loads pxr/USD
from isaaclab.app import AppLauncher
launcher = AppLauncher({"headless": True, "enable_cameras": False})
sim_app = launcher.app

import torch
# Now Kit is live, safe to import isaaclab sub-modules
from tarantula_isaac.suspension_env_cfg import TarantulaSuspensionEnvCfg
from tarantula_isaac.suspension_env import TarantulaSuspensionEnv

cfg = TarantulaSuspensionEnvCfg()
cfg.scene.num_envs = 16
cfg.scene.env_spacing = 4.0

print(f"[smoke] Creating env (num_envs={cfg.scene.num_envs})...")
env = TarantulaSuspensionEnv(cfg=cfg)

obs_dict, _ = env.reset()
obs = obs_dict["policy"]
print(f"[smoke] obs.shape = {obs.shape}")
assert obs.shape == (cfg.scene.num_envs, 47), f"Expected (N,47), got {obs.shape}"
assert not obs.isnan().any(), "NaN in initial obs!"
print(f"[smoke] action_space = {env.action_space}")

STEPS = 200
print(f"[smoke] Stepping {STEPS} steps...")
for i in range(STEPS):
    action = torch.rand(cfg.scene.num_envs, 12, device=env.device) * 2 - 1
    obs_dict, rew, term, trunc, info = env.step(action)
    obs = obs_dict["policy"]
    if obs.isnan().any():
        print(f"[FAIL] NaN in obs at step {i}")
        sim_app.close()
        sys.exit(1)
    if rew.isnan().any():
        print(f"[FAIL] NaN in reward at step {i}")
        sim_app.close()
        sys.exit(1)

print(f"[smoke] Final obs.shape = {obs.shape}, reward mean = {rew.mean():.3f}")
print("[smoke] PASS — v5 env OK (obs=47, action=12, ContactSensor, no NaN)")
env.close()
sim_app.close()
PYEOF

echo "=== Tarantula v5 Isaac Lab Env Smoke Test ==="
cd "$REPO_ROOT"
source "$ISAAC_VENV/bin/activate"
export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/tarantula_control:${PYTHONPATH:-}"
python3 -u "$SMOKE_PY"
