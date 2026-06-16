#!/bin/bash
# Isaac Lab headless smoke check for the shared Gazebo/Isaac heightmap terrain.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAAC_VENV="${ISAAC_VENV:-/home/ang/isaac_venv}"
TERRAIN_DIR="${TERRAIN_DIR:-${REPO_ROOT}/generated/terrains/gazebo_demo/42}"
NUM_ENVS="${NUM_ENVS:-2}"
LOG=/tmp/isaac_shared_terrain_smoke.log
LIMIT_KB=$((12*1024*1024))

source "$ISAAC_VENV/bin/activate"
export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/tarantula_control:${PYTHONPATH:-}"

python3 -u "${REPO_ROOT}/src/tarantula_isaac/shared_terrain_smoke.py" \
  --headless \
  --terrain-dir "$TERRAIN_DIR" \
  --num-envs "$NUM_ENVS" \
  > "$LOG" 2>&1 &
PID=$!

for _ in $(seq 1 300); do
  kill -0 $PID 2>/dev/null || break
  RSS=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
  if [ -n "$RSS" ] && [ "$RSS" -gt "$LIMIT_KB" ]; then
    echo "WATCHDOG: RSS ${RSS}KB exceeded ${LIMIT_KB}KB, killing $PID" >> "$LOG"
    kill -9 $PID
    break
  fi
  sleep 1
done

wait $PID 2>/dev/null
grep -E "TERRAIN_IMPORTER|ENV_ORIGINS|SHARED_TERRAIN_SMOKE_OK|Traceback|Error|WATCHDOG" "$LOG"
