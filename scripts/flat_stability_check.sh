#!/bin/bash
# 最小隔离测试：平地出生，记录落地后 12 秒车身倾角
WS=~/tarantula_ws
source /opt/ros/humble/setup.bash
source $WS/install/setup.bash
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)

ros2 launch tarantula_bringup sim.launch.py gui:=false > /tmp/sim_flat.log 2>&1 &
LAUNCH_PID=$!
for i in $(seq 1 30); do
  [ "$(ros2 control list_controllers 2>/dev/null | grep -c active)" -ge 3 ] && break
  sleep 2
done

python3 - <<'EOF'
import rclpy, math, time
from rclpy.node import Node
from sensor_msgs.msg import Imu
rclpy.init()
node = Node('tilt_logger')
buf = []
def cb(m):
    q = m.orientation
    tilt = math.degrees(math.acos(max(-1, min(1, 1 - 2*(q.x*q.x + q.y*q.y)))))
    buf.append((time.time(), tilt))
node.create_subscription(Imu, '/imu/data', cb, 50)
t0 = time.time(); last = t0
while time.time() - t0 < 45:
    rclpy.spin_once(node, timeout_sec=0.2)
    if time.time() - last >= 1.0 and buf:
        sec = [b[1] for b in buf if b[0] > last]
        if sec:
            print(f"t={time.time()-t0:4.1f}s  tilt={sum(sec)/len(sec):6.2f} deg", flush=True)
        last = time.time()
rclpy.shutdown()
EOF

kill $LAUNCH_PID 2>/dev/null
sleep 3
pkill -f gzserver 2>/dev/null; pkill -f active_suspension 2>/dev/null
echo "=== done ==="
