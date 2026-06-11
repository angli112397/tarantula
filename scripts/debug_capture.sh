#!/bin/bash
# 平地极限环诊断：渐入完成后采集 /active_suspension/debug 15s，分析振荡
WS=~/tarantula_ws
source /opt/ros/humble/setup.bash
source $WS/install/setup.bash
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)

ros2 launch tarantula_bringup sim.launch.py gui:=false > /tmp/sim_dbg.log 2>&1 &
LP=$!
for i in $(seq 1 30); do
  [ "$(ros2 control list_controllers 2>/dev/null | grep -c active)" -ge 3 ] && break
  sleep 2
done
sleep 25  # 落地保持 + 渐入 + 进入"稳态"

python3 - <<'EOF'
import rclpy, math, time
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
rclpy.init()
node = Node('dbg_sampler')
rows = []
node.create_subscription(Float64MultiArray, '/active_suspension/debug',
                         lambda m: rows.append(list(m.data)), 50)
end = time.time() + 15
while time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.2)
import statistics as st
def stat(idx, name):
    v = [r[idx] for r in rows]
    print(f"{name:12s} mean={st.mean(v):+.4f} std={st.pstdev(v):.4f} "
          f"min={min(v):+.4f} max={max(v):+.4f}")
print(f"samples: {len(rows)}  (字段: roll pitch u_roll u_pitch qt_fl qt_mr q_fl tau_fl)")
for idx, name in [(0,'roll'),(1,'pitch'),(2,'u_roll'),(3,'u_pitch'),
                  (4,'qt_fl'),(7,'qt_mr'),(10,'q_fl'),(16,'tau_fl')]:
    stat(idx, name)
# 估算 roll 振荡主周期（过零间隔）
v = [r[0] for r in rows]
m = st.mean(v)
crossings = [i for i in range(1, len(v)) if (v[i-1]-m)*(v[i]-m) < 0]
if len(crossings) > 3:
    import itertools
    gaps = [b-a for a, b in zip(crossings, crossings[1:])]
    print(f"roll 过零 {len(crossings)} 次，平均半周期 ≈ {st.mean(gaps)*0.1:.2f}s (10Hz 采样)")
EOF

kill $LP 2>/dev/null
sleep 3
pkill -f gzserver 2>/dev/null; pkill -f active_suspension 2>/dev/null
echo "=== done ==="
