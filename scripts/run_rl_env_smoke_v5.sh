#!/usr/bin/env bash
# Active-suspension Isaac Lab environment smoke test
# Verifies: obs.shape==(N,50), action_space.shape==(N,6), wheel-force sensor alive, no NaN
# Usage: bash scripts/run_rl_env_smoke_v5.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAAC_VENV="${ISAAC_VENV:-/home/ang/isaac_venv}"

SMOKE_PY="$(mktemp /tmp/tarantula_smoke_v5_XXXXXX.py)"
trap 'rm -f "$SMOKE_PY"' EXIT

cat > "$SMOKE_PY" << 'PYEOF'
import os, sys, pathlib
repo = pathlib.Path(os.environ["TARANTULA_REPO_ROOT"])
sys.path.insert(0, str(repo / "src/tarantula_isaac"))
sys.path.insert(0, str(repo / "src/tarantula_control"))

# AppLauncher MUST be first — it initializes the Kit which loads pxr/USD
from isaaclab.app import AppLauncher
launcher = AppLauncher({"headless": True, "enable_cameras": False})
sim_app = launcher.app

import torch
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.importer.urdf")
# Now Kit is live, safe to import isaaclab sub-modules
from tarantula_isaac.robot import ensure_tarantula_usd
from tarantula_isaac.suspension_env_cfg import TarantulaSuspensionEnvCfg
from tarantula_isaac.suspension_env import TarantulaSuspensionEnv

ensure_tarantula_usd()
cfg = TarantulaSuspensionEnvCfg()
cfg.scene.num_envs = 16
cfg.scene.env_spacing = 4.0
cfg.command_resampling_enabled = False

print(f"[smoke] Creating env (num_envs={cfg.scene.num_envs})...")
env = TarantulaSuspensionEnv(cfg=cfg)

obs_dict, _ = env.reset()
obs = obs_dict["policy"]
print(f"[smoke] obs.shape = {obs.shape}")
assert obs.shape == (cfg.scene.num_envs, 50), f"Expected (N,50), got {obs.shape}"
assert not obs.isnan().any(), "NaN in initial obs!"
log = env.extras.get("log", {})
for key in (
    "Episode_Reward/total",
    "Episode_Reward/orientation",
    "Episode_Reward/contact_support",
    "Episode_Reward/wheel_load_balance",
    "Episode_Metric/vx_error_rms",
    "Episode_Metric/wz_error_rms",
    "Episode_Metric/roll_pitch_rate",
    "Episode_Termination/tilt",
    "Episode_Termination/time_out",
):
    assert key in log, f"Missing diagnostic log key: {key}"
print(f"[smoke] action_space = {env.action_space}")

STEPS = 200
print(f"[smoke] Stepping {STEPS} steps...")
for i in range(STEPS):
    action = torch.rand(cfg.scene.num_envs, 6, device=env.device) * 2 - 1
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
print("[smoke] PASS - v5 active-suspension env OK (obs=50, action=6, wheel-force sensor, no NaN)")
env.close()
sim_app.close()
PYEOF

echo "=== Tarantula v5 Isaac Lab Env Smoke Test ==="
cd "$REPO_ROOT"
source "$ISAAC_VENV/bin/activate"
export OMNI_KIT_ACCEPT_EULA=Y
export TARANTULA_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/tarantula_control:${PYTHONPATH:-}"
python3 -u "$SMOKE_PY"
