#!/bin/bash
# Isaac Lab headless regression check: chassis settles with wheels on the
# ground (SPAWN_Z_OFFSET matches the chassis URDF) and drives at low cmd_vx.
# Rerun this after any tarantula_chassis_v2.xacro, tarantula_common.xacro, or robot.py change.
set -uo pipefail
source ~/isaac_venv/bin/activate
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/tarantula_control:${PYTHONPATH:-}"

LOG=/tmp/isaac_geometry_check.log
python3 -u "${REPO_ROOT}/src/tarantula_isaac/geometry_check.py" > "$LOG" 2>&1 &
PID=$!
LIMIT_KB=$((12*1024*1024))  # 12GB RSS safety cap (15GB RAM + 15GB swap)
for i in $(seq 1 300); do
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
grep -E "^\[(settle|drive)\]|GEOMETRY_CHECK_OK|AssertionError|WATCHDOG" "$LOG"
