#!/usr/bin/env python3
"""自动化导航任务：把 world 系目标经 T_map_world 变换后顺序下发。

用法: nav_mission.py x1,y1,yaw1 x2,y2,yaw2 ...
前提: sim.launch.py 与 nav.launch.py 均已运行（map->odom TF 在发布）。

注意:
  - SLAM 的 map 系锚定在启动时刻的里程计原点，与 Gazebo world 系存在
    偏差且随滑移漂移，自动化测试必须经 tf 变换下发目标（RViz 点目标
    无此问题）；变换以当前真值+tf 采样，发送前机器人应静止；
  - 目标点选平地，勿压台阶沿（实测压沿时机器人在 0.12m 台阶边缘
    无法满足 0.25m 到位判据）。
"""
import math
import subprocess
import sys
import time

import rclpy
import tf2_ros
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


def truth_pose():
    out = subprocess.check_output(['gz', 'model', '-m', 'tarantula', '-p'],
                                  text=True).strip().split('\n')[-1].split()
    return float(out[0]), float(out[1]), float(out[5])


def main():
    goals = [tuple(float(v) for v in arg.split(',')) for arg in sys.argv[1:]]
    rclpy.init()
    n = Node('nav_mission')
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, n)
    t0 = time.time()
    while time.time() - t0 < 30:
        rclpy.spin_once(n, timeout_sec=0.2)
        if buf.can_transform('map', 'base_link', rclpy.time.Time()):
            break
    else:
        print('FATAL: no map->base_link tf')
        return

    # T_map_world = T_map_base * inv(T_world_base)，机器人静止时采样
    wx, wy, wyaw = truth_pose()
    tr = buf.lookup_transform('map', 'base_link', rclpy.time.Time())
    q = tr.transform.rotation
    myaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))
    mx, my = tr.transform.translation.x, tr.transform.translation.y
    dyaw = myaw - wyaw
    c, s = math.cos(dyaw), math.sin(dyaw)
    tx, ty = mx - (c * wx - s * wy), my - (s * wx + c * wy)
    print(f'T_map_world: dyaw={dyaw:.3f} t=({tx:.3f},{ty:.3f})', flush=True)

    client = ActionClient(n, NavigateToPose, 'navigate_to_pose')
    client.wait_for_server()
    for gx, gy, gyaw in goals:
        goal = NavigateToPose.Goal()
        p = goal.pose
        p.header.frame_id = 'map'
        p.pose.position.x = c * gx - s * gy + tx
        p.pose.position.y = s * gx + c * gy + ty
        myaw_g = gyaw + dyaw
        p.pose.orientation.z = math.sin(myaw_g / 2)
        p.pose.orientation.w = math.cos(myaw_g / 2)
        print(f'goal world ({gx},{gy}) -> map '
              f'({p.pose.position.x:.2f},{p.pose.position.y:.2f})', flush=True)
        fut = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(n, fut)
        res_fut = fut.result().get_result_async()
        rclpy.spin_until_future_complete(n, res_fut)
        status = res_fut.result().status  # 4=SUCCEEDED 6=ABORTED
        twx, twy, twyaw = truth_pose()
        name = {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}.get(status, status)
        print(f'RESULT {name} | truth ({twx:.2f},{twy:.2f},yaw {twyaw:.2f}) '
              f'| err {math.hypot(twx - gx, twy - gy):.2f} m', flush=True)
        if status != 4:
            break
    rclpy.shutdown()


if __name__ == '__main__':
    main()
