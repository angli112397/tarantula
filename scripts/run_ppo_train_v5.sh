#!/usr/bin/env bash
# Active-suspension PPO training (obs=50, action=6).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAAC_VENV="${ISAAC_VENV:-/home/ang/isaac_venv}"
TRAIN_PY="${REPO_ROOT}/src/tarantula_isaac/train_v5.py"

NUM_ENVS="${NUM_ENVS:-64}"
LOG=/tmp/tarantula_ppo_v5.log
LIMIT_KB=$((12*1024*1024))  # 12 GB RSS watchdog

echo "=== Tarantula PPO Training (active suspension posture) ==="
echo "    num_envs=${NUM_ENVS}, log=${LOG}"
echo "    extra_args: $*"

source "$ISAAC_VENV/bin/activate"
export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/tarantula_control:${PYTHONPATH:-}"

python3 -u "$TRAIN_PY" \
  --num_envs "$NUM_ENVS" \
  "$@" \
  > "$LOG" 2>&1 &
PID=$!

echo "Training PID=$PID, watching RSS..."
for i in $(seq 1 3600); do
  sleep 5
  kill -0 $PID 2>/dev/null || { echo "Training process exited."; break; }
  RSS=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
  if [ -n "$RSS" ] && [ "$RSS" -gt "$LIMIT_KB" ]; then
    echo "WATCHDOG: RSS ${RSS}KB > ${LIMIT_KB}KB, killing $PID"
    kill -9 $PID
    break
  fi
  # Print last reward line every 60s
  if (( i % 12 == 0 )); then
    tail -n8 "$LOG" | grep -E "Iteration|Mean reward|reward|Episode_Metric" | tail -1 || true
  fi
done

wait $PID
echo "=== Training done. Log: $LOG ==="
