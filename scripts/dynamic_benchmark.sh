#!/bin/bash
# G3 验收：崎岖地形 0.8 m/s 行驶，主动调平 vs 纯被动的 roll/pitch RMS 对照
# （两次独立冷启动）。判据：主动相 RMS 比被动降低 >= 40%。
# 注意：不要加 set -u，ROS 的 setup.bash 含未定义变量
WS=~/tarantula_ws
source /opt/ros/humble/setup.bash
source $WS/install/setup.bash
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)

drive_and_measure() {  # 行驶 16s 并统计姿态 RMS
  timeout 16 ros2 topic pub -r 10 /diff_drive_controller/cmd_vel_unstamped \
    geometry_msgs/msg/Twist "{linear: {x: 0.8}}" > /dev/null 2>&1 &
  python3 - <<'EOF'
import rclpy, math, time
from rclpy.node import Node
from sensor_msgs.msg import Imu
rclpy.init()
node = Node('rms_sampler')
rolls, pitches = [], []
def cb(m):
    q = m.orientation
    sr = 2*(q.w*q.x + q.y*q.z); cr = 1-2*(q.x*q.x+q.y*q.y)
    rolls.append(math.atan2(sr, cr))
    pitches.append(math.asin(max(-1, min(1, 2*(q.w*q.y - q.z*q.x)))))
node.create_subscription(Imu, '/imu/data', cb, 50)
end = time.time() + 16
while time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.2)
def rms(v): return math.degrees(math.sqrt(sum(x*x for x in v)/len(v))) if v else 0.0
def mx(v): return math.degrees(max(abs(x) for x in v)) if v else 0.0
print(f"RMS roll={rms(rolls):.2f} deg, RMS pitch={rms(pitches):.2f} deg | "
      f"max roll={mx(rolls):.2f}, max pitch={mx(pitches):.2f} ({len(rolls)} samples)")
rclpy.shutdown()
EOF
}

run_phase() {  # $1=leveling true/false  $2=标题
  echo "=== $2 ==="
  ros2 launch tarantula_bringup sim.launch.py gui:=false leveling:=$1 > /tmp/sim_dyn_$1.log 2>&1 &
  local LP=$!
  for i in $(seq 1 30); do
    [ "$(ros2 control list_controllers 2>/dev/null | grep -c active)" -ge 3 ] && break
    sleep 2
  done
  sleep 18  # 落地保持期(5s sim) + 增益渐入，RTF 余量
  drive_and_measure
  kill $LP 2>/dev/null
  sleep 4
  pkill -f gzserver 2>/dev/null; pkill -f active_suspension 2>/dev/null
  sleep 3
}

run_phase true  "[A] 主动调平 ON（崎岖路 0.8 m/s）"
run_phase false "[B] 纯被动悬挂（崎岖路 0.8 m/s）"
echo "=== done ==="
