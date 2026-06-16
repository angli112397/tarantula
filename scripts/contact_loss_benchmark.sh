#!/bin/bash
# M1 验收：崎岖路低速行驶，统计六腿接触丢失时长（来自 ~/debug 的 contact×6），
# 同时记录 roll/pitch RMS/max 与 yaw 漂移（自旋/翻车检查）。
# 速度 0.2 m/s（用户观察 0.8 m/s 在 bump_2 处因左右轮交替冲击产生大幅转向，
# 偏离航线导致后续地形特征未被覆盖；0.1-0.3 m/s 也更贴近 SLAM/Nav2 目标速度，
# 且便于肉眼观察姿态细节）。行驶时长按比例放大以覆盖相近的路程。
# 用法：contact_loss_benchmark.sh <label> "<标题>"
#   label 用于区分日志文件名（如 off / on）
#   可用环境变量 SPEED / DURATION 覆盖默认值
# 注意：不要加 set -u，ROS 的 setup.bash 含未定义变量
WS=~/tarantula_ws
source /opt/ros/humble/setup.bash
source $WS/install/setup.bash
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)

SPEED=${SPEED:-0.2}
DURATION=${DURATION:-60}
export SPEED DURATION

drive_and_measure() {  # 行驶 $DURATION 秒，统计姿态 RMS/yaw 漂移 与 六腿接触丢失时长
  timeout $DURATION ros2 topic pub -r 10 /diff_drive_controller/cmd_vel_unstamped \
    geometry_msgs/msg/Twist "{linear: {x: $SPEED}}" > /dev/null 2>&1 &
  python3 - <<'EOF'
import rclpy, math, time, os
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray

DURATION = float(os.environ.get('DURATION', '16'))

LEGS = ['fl', 'fr', 'ml', 'mr', 'rl', 'rr']
rclpy.init()
node = Node('contact_sampler')

rolls, pitches, yaws = [], [], []
def imu_cb(m):
    q = m.orientation
    sr = 2*(q.w*q.x + q.y*q.z); cr = 1-2*(q.x*q.x+q.y*q.y)
    rolls.append(math.atan2(sr, cr))
    pitches.append(math.asin(max(-1, min(1, 2*(q.w*q.y - q.z*q.x)))))
    sy = 2*(q.w*q.z + q.x*q.y); cy = 1-2*(q.y*q.y+q.z*q.z)
    yaws.append(math.atan2(sy, cy))
node.create_subscription(Imu, '/imu/data', imu_cb, 50)

contact_samples = {leg: [] for leg in LEGS}
def dbg_cb(m):
    d = m.data
    if len(d) < 28:
        return
    for i, leg in enumerate(LEGS):
        contact_samples[leg].append(d[22 + i])
node.create_subscription(Float64MultiArray, '/active_suspension/debug', dbg_cb, 50)

end = time.time() + DURATION
while time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.2)

def rms(v): return math.degrees(math.sqrt(sum(x*x for x in v)/len(v))) if v else 0.0
def mx(v): return math.degrees(max(abs(x) for x in v)) if v else 0.0

print(f"RMS roll={rms(rolls):.2f} deg, RMS pitch={rms(pitches):.2f} deg | "
      f"max roll={mx(rolls):.2f}, max pitch={mx(pitches):.2f} ({len(rolls)} samples)")
yaw0 = math.degrees(yaws[0]) if yaws else 0.0
yawN = math.degrees(yaws[-1]) if yaws else 0.0
print(f"yaw drift = {yawN - yaw0:+.2f} deg (start {yaw0:+.2f} -> end {yawN:+.2f})")

dt = 0.1  # ~/debug 每 10 个控制步发一次 @ 100Hz = 10Hz
total_lost = 0.0
for leg in LEGS:
    s = contact_samples[leg]
    lost = sum(1 for c in s if c < 0.5) * dt
    total_lost += lost
    dur = len(s) * dt
    pct = (100 * lost / dur) if dur > 0 else 0.0
    print(f"  {leg}: lost={lost:.2f}s / {dur:.2f}s ({pct:.1f}%)")
print(f"TOTAL contact-loss time (sum over 6 legs) = {total_lost:.2f}s")
rclpy.shutdown()
EOF
}

run_phase() {  # $1=label（日志文件名）  $2=标题
  echo "=== $2 ==="
  ros2 launch tarantula_bringup sim.launch.py leveling:=true > /tmp/sim_m1_$1.log 2>&1 &
  local LP=$!
  for i in $(seq 1 30); do
    [ "$(ros2 control list_controllers 2>/dev/null | grep -c active)" -ge 3 ] && break
    sleep 2
  done
  sleep 18  # 落地保持期(5s sim) + 增益渐入，RTF 余量
  drive_and_measure
  kill $LP 2>/dev/null
  sleep 4
  pkill -f "ign gazebo" 2>/dev/null; pkill -f active_suspension 2>/dev/null
  sleep 3
}

run_phase "$1" "$2"
echo "=== done ==="
