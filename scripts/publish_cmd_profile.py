#!/usr/bin/env python3
"""Publish a reusable cmd_vel profile to a running ROS/Gazebo session."""

from __future__ import annotations

import argparse
import time

try:
    from tarantula_control.command_profiles import PROFILE_CHOICES, parse_route_specs
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError("tarantula_control modules are not available. Run `source install/setup.bash` first.") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a command profile to /cmd_vel.")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="navi")
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="Optional custom segment: name,vx,wz[,duration_s]. May be repeated.",
    )
    parser.add_argument("--default-duration", type=float, default=4.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved profile and exit.")
    args = parser.parse_args()

    sequence = parse_route_specs(args.segment, profile=args.profile, default_duration_s=args.default_duration)
    if args.dry_run:
        for segment in sequence:
            print(f"{segment.name}: vx={segment.vx:.3f}, wz={segment.wz:.3f}, duration={segment.duration_s:.3f}s")
        print(f"total_duration={sum(segment.duration_s for segment in sequence):.3f}s")
        return 0

    try:
        import rclpy
        from geometry_msgs.msg import Twist
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("ROS2 Python modules are not available. Run `source /opt/ros/humble/setup.bash` first.") from exc

    rclpy.init()
    node = rclpy.create_node("cmd_profile_publisher")
    publisher = node.create_publisher(Twist, args.topic, 10)
    period_s = 1.0 / args.rate
    try:
        time.sleep(0.5)
        for segment in sequence:
            node.get_logger().info(
                f"publishing {segment.name}: vx={segment.vx:.3f}, wz={segment.wz:.3f}, duration={segment.duration_s:.2f}s"
            )
            deadline = time.monotonic() + segment.duration_s
            while time.monotonic() < deadline:
                msg = Twist()
                msg.linear.x = segment.vx
                msg.angular.z = segment.wz
                publisher.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(period_s)
        publisher.publish(Twist())
        node.get_logger().info("profile complete; published final stop")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
