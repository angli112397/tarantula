#!/bin/bash
# G2 验收：8 度侧倾斜面，主动调平 vs 纯被动悬挂（两次独立冷启动，避免热切换
# 时悬挂储能释放掀翻车身——2026-06-11 实验教训）。
# 注意：不要加 set -u，ROS 的 setup.bash 含未定义变量
WS=~/tarantula_ws
source /opt/ros/humble/setup.bash
source $WS/install/setup.bash
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)
WORLD=$WS/install/tarantula_bringup/share/tarantula_bringup/worlds/tilt_test.world

log_tilt() {  # $1=时长秒，逐秒打印倾角/偏航，最后打印末 5 个秒均值的平均
  python3 - "$1" <<'EOF'
import rclpy, math, time, sys
from rclpy.node import Node
from sensor_msgs.msg import Imu
dur = float(sys.argv[1])
rclpy.init()
node = Node('tilt_logger')
buf, per_sec = [], []
def cb(m):
    q = m.orientation
    tilt = math.degrees(math.acos(max(-1, min(1, 1 - 2*(q.x*q.x + q.y*q.y)))))
    yaw = math.degrees(math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))
    buf.append((time.time(), tilt, yaw))
node.create_subscription(Imu, '/imu/data', cb, 50)
t0 = time.time(); last = t0
while time.time() - t0 < dur:
    rclpy.spin_once(node, timeout_sec=0.2)
    if time.time() - last >= 1.0 and buf:
        sec = [b for b in buf if b[0] > last]
        if sec:
            m_tilt = sum(s[1] for s in sec)/len(sec)
            per_sec.append(m_tilt)
            print(f"t={time.time()-t0:4.1f}s  tilt={m_tilt:5.2f} deg  yaw={sec[-1][2]:+6.1f} deg", flush=True)
        last = time.time()
if per_sec:
    tail = per_sec[-5:]
    print(f">>> 末 {len(tail)} 秒平均倾角: {sum(tail)/len(tail):.2f} deg")
rclpy.shutdown()
EOF
}

run_phase() {  # $1=leveling true/false  $2=时长  $3=标题
  echo "=== $3 ==="
  ros2 launch tarantula_bringup sim.launch.py gui:=false world:=$WORLD \
    spawn_z:=0.84 leveling:=$1 > /tmp/sim_tilt_$1.log 2>&1 &
  local LP=$!
  for i in $(seq 1 30); do
    [ "$(ros2 control list_controllers 2>/dev/null | grep -c active)" -ge 3 ] && break
    sleep 2
  done
  log_tilt $2
  kill $LP 2>/dev/null
  sleep 4
  pkill -f "ign gazebo" 2>/dev/null; pkill -f active_suspension 2>/dev/null
  sleep 3
}

run_phase true  55 "[A] 主动调平 ON（斜面 8 度）"
run_phase false 12 "[B] 纯被动悬挂（独立冷启动）"
echo "=== done ==="
