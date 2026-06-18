#!/usr/bin/env python3
"""Minimal frontier exploration driver for online SLAM + Nav2 demos.

The node reads the current /map, finds unknown cells adjacent to free cells,
clusters them as frontiers, and sends conservative free-space goals through
Nav2's NavigateToPose action. It is intentionally small and demo-oriented:
Nav2 remains responsible for feasibility, obstacle avoidance, and recovery.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener


FREE_THRESHOLD = 20
OCCUPIED_THRESHOLD = 65


@dataclass(frozen=True)
class GridPoint:
    i: int
    j: int


@dataclass(frozen=True)
class Candidate:
    goal_x: float
    goal_y: float
    yaw: float
    score: float
    frontier_cells: int
    clearance: float


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class FrontierExplorer(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__(
            "nav2_frontier_explorer",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
            automatically_declare_parameters_from_overrides=True,
        )
        self.args = args
        self.map_msg: OccupancyGrid | None = None
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGrid, "/map", self._on_map, map_qos)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.planner_client = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")
        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.blacklist: list[tuple[float, float]] = []
        self._last_tf_warning = 0.0

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg

    def robot_xy(self) -> tuple[float, float] | None:
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except TransformException as exc:
            now = time.monotonic()
            if now - self._last_tf_warning > 2.0:
                self.get_logger().warn(f"waiting for map->base_link: {exc}")
                self._last_tf_warning = now
            return None
        return tf.transform.translation.x, tf.transform.translation.y

    def wait_until_ready(self) -> bool:
        end = time.monotonic() + self.args.ready_timeout
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.map_msg is None:
                continue
            if self.robot_xy() is None:
                continue
            planner_ready = self.planner_client.wait_for_server(timeout_sec=0.1)
            nav_ready = self.nav_client.wait_for_server(timeout_sec=0.1)
            if planner_ready and nav_ready:
                return True
        return False

    def world_xy(self, point: GridPoint) -> tuple[float, float]:
        assert self.map_msg is not None
        info = self.map_msg.info
        return (
            info.origin.position.x + (point.j + 0.5) * info.resolution,
            info.origin.position.y + (point.i + 0.5) * info.resolution,
        )

    def is_blacklisted(self, x: float, y: float) -> bool:
        radius2 = self.args.blacklist_radius * self.args.blacklist_radius
        return any((x - bx) ** 2 + (y - by) ** 2 < radius2 for bx, by in self.blacklist)

    def find_candidates(self) -> list[Candidate]:
        if self.map_msg is None:
            return []
        robot = self.robot_xy()
        if robot is None:
            return []

        info = self.map_msg.info
        width = info.width
        height = info.height
        data = self.map_msg.data
        clearance_cells = max(1, int(round(self.args.min_clearance / info.resolution)))

        def idx(i: int, j: int) -> int:
            return i * width + j

        def inside(i: int, j: int) -> bool:
            return 0 <= i < height and 0 <= j < width

        def free(i: int, j: int) -> bool:
            return inside(i, j) and 0 <= data[idx(i, j)] <= FREE_THRESHOLD

        def unknown(i: int, j: int) -> bool:
            return inside(i, j) and data[idx(i, j)] < 0

        def occupied(i: int, j: int) -> bool:
            return inside(i, j) and data[idx(i, j)] >= OCCUPIED_THRESHOLD

        obstacle_distance = [[10**9 for _ in range(width)] for _ in range(height)]
        obstacle_queue: deque[GridPoint] = deque()
        for i in range(height):
            for j in range(width):
                if occupied(i, j):
                    obstacle_distance[i][j] = 0
                    obstacle_queue.append(GridPoint(i, j))
        while obstacle_queue:
            point = obstacle_queue.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = point.i + di, point.j + dj
                if inside(ni, nj) and obstacle_distance[ni][nj] > obstacle_distance[point.i][point.j] + 1:
                    obstacle_distance[ni][nj] = obstacle_distance[point.i][point.j] + 1
                    obstacle_queue.append(GridPoint(ni, nj))

        def near_occupied(i: int, j: int) -> bool:
            return obstacle_distance[i][j] < clearance_cells

        frontiers: set[GridPoint] = set()
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if not unknown(i, j):
                    continue
                if any(free(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    frontiers.add(GridPoint(i, j))

        candidates: list[Candidate] = []
        visited: set[GridPoint] = set()
        for start in frontiers:
            if start in visited:
                continue
            cluster: list[GridPoint] = []
            queue = deque([start])
            visited.add(start)
            while queue:
                point = queue.popleft()
                cluster.append(point)
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = GridPoint(point.i + di, point.j + dj)
                    if nxt in frontiers and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)

            if len(cluster) < self.args.min_cluster_cells:
                continue

            cluster_frontier = set(cluster)
            standoff_cells = max(1, int(round(self.args.frontier_standoff / info.resolution)))
            nearby_free: set[GridPoint] = set()
            for point in cluster:
                for di in range(-standoff_cells, standoff_cells + 1):
                    for dj in range(-standoff_cells, standoff_cells + 1):
                        if di * di + dj * dj > standoff_cells * standoff_cells:
                            continue
                        ni, nj = point.i + di, point.j + dj
                        if free(ni, nj) and not near_occupied(ni, nj):
                            nearby_free.add(GridPoint(ni, nj))
            if not nearby_free:
                continue

            frontier_x = sum(self.world_xy(point)[0] for point in cluster) / len(cluster)
            frontier_y = sum(self.world_xy(point)[1] for point in cluster) / len(cluster)

            def frontier_grid_distance(point: GridPoint) -> int:
                return min(abs(point.i - f.i) + abs(point.j - f.j) for f in cluster_frontier)

            def goal_score(point: GridPoint) -> float:
                x, y = self.world_xy(point)
                robot_dist = math.hypot(x - robot[0], y - robot[1])
                frontier_dist = frontier_grid_distance(point) * info.resolution
                clearance = obstacle_distance[point.i][point.j] * info.resolution
                return (
                    robot_dist
                    + self.args.frontier_distance_weight * frontier_dist
                    - self.args.clearance_gain * clearance
                )

            best_goal = min(nearby_free, key=goal_score)
            goal_x, goal_y = self.world_xy(best_goal)
            if self.is_blacklisted(goal_x, goal_y):
                continue

            dist = math.hypot(goal_x - robot[0], goal_y - robot[1])
            if dist < self.args.min_goal_distance:
                continue
            yaw = math.atan2(frontier_y - goal_y, frontier_x - goal_x)
            clearance = obstacle_distance[best_goal.i][best_goal.j] * info.resolution
            score = (
                dist
                - self.args.cluster_gain * math.sqrt(len(cluster))
                - self.args.clearance_gain * clearance
            )
            candidates.append(Candidate(goal_x, goal_y, yaw, score, len(cluster), clearance))

        if not candidates:
            return []
        candidates.sort(key=lambda c: c.score)
        return candidates

    def make_goal_pose(self, candidate: Candidate) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = candidate.goal_x
        pose.pose.position.y = candidate.goal_y
        pose.pose.orientation = yaw_to_quaternion(candidate.yaw)
        return pose

    def has_valid_plan(self, candidate: Candidate) -> bool:
        goal = ComputePathToPose.Goal()
        goal.goal = self.make_goal_pose(candidate)
        goal.planner_id = "GridBased"
        goal.use_start = False

        send_future = self.planner_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=self.args.plan_timeout)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.args.plan_timeout)
        if not result_future.done():
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=1.0)
            return False

        wrapped = result_future.result()
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            return False
        return len(wrapped.result.path.poses) > 1

    def send_goal(self, candidate: Candidate) -> int:
        goal = NavigateToPose.Goal()
        goal.pose = self.make_goal_pose(candidate)

        self.get_logger().info(
            "frontier goal x=%.2f y=%.2f yaw=%.2f cells=%d score=%.2f"
            % (candidate.goal_x, candidate.goal_y, candidate.yaw, candidate.frontier_cells, candidate.score)
        )
        send_future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().warn("frontier goal rejected")
            return GoalStatus.STATUS_ABORTED

        result_future = handle.get_result_async()
        deadline = time.monotonic() + self.args.goal_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if result_future.done():
                status = result_future.result().status
                self.get_logger().info(f"frontier goal finished with status={status}")
                return status

        self.get_logger().warn("frontier goal timed out; canceling")
        cancel_future = handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
        return GoalStatus.STATUS_CANCELED

    def run(self) -> int:
        if not self.wait_until_ready():
            self.get_logger().error("explorer not ready: missing /map, map->base_link, or Nav2 action server")
            return 2

        successes = 0
        for _ in range(self.args.max_goals):
            candidates = self.find_candidates()
            if not candidates:
                self.get_logger().info("no usable frontier candidates remain")
                break
            candidate = None
            for option in candidates[: self.args.candidate_attempts]:
                if self.has_valid_plan(option):
                    candidate = option
                    break
                self.get_logger().info(
                    "blacklisting unreachable frontier x=%.2f y=%.2f cells=%d clearance=%.2f"
                    % (option.goal_x, option.goal_y, option.frontier_cells, option.clearance)
                )
                self.blacklist.append((option.goal_x, option.goal_y))
            if candidate is None:
                self.get_logger().info("no frontier candidate passed Nav2 planning precheck")
                break
            status = self.send_goal(candidate)
            if status == GoalStatus.STATUS_SUCCEEDED:
                successes += 1
                continue
            self.blacklist.append((candidate.goal_x, candidate.goal_y))
            if self.args.stop_on_failure:
                break
        self.get_logger().info(f"frontier exploration finished: successes={successes}")
        return 0 if successes > 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-goals", type=int, default=8)
    parser.add_argument("--goal-timeout", type=float, default=90.0)
    parser.add_argument("--plan-timeout", type=float, default=6.0)
    parser.add_argument("--ready-timeout", type=float, default=20.0)
    parser.add_argument("--min-cluster-cells", type=int, default=8)
    parser.add_argument("--min-clearance", type=float, default=0.75)
    parser.add_argument("--min-goal-distance", type=float, default=1.2)
    parser.add_argument("--frontier-standoff", type=float, default=2.8)
    parser.add_argument("--frontier-distance-weight", type=float, default=0.5)
    parser.add_argument("--clearance-gain", type=float, default=0.6)
    parser.add_argument("--blacklist-radius", type=float, default=0.8)
    parser.add_argument("--candidate-attempts", type=int, default=30)
    parser.add_argument("--cluster-gain", type=float, default=0.03)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init(args=None)
    node = FrontierExplorer(args)
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
