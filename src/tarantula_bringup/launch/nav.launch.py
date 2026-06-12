import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """SLAM 在线建图 + Nav2 导航（先 ros2 launch tarantula_bringup sim.launch.py）。

    自建最小 Nav2 bringup 而非 include nav2_bringup：需要把 cmd_vel
    remap 到 /diff_drive_controller/cmd_vel_unstamped（controller_server
    和 behavior_server 都直接发速度指令），nav2_bringup 不暴露该 remap，
    保持机器人侧接口不变、也不引入 relay 节点。
    map->odom 由 slam_toolbox 提供，无 AMCL/map_server。
    """
    bringup_dir = get_package_share_directory('tarantula_bringup')
    params_file = LaunchConfiguration('params_file')
    cmd_vel_remap = ('cmd_vel', '/diff_drive_controller/cmd_vel_unstamped')

    # 姿态门控扫描过滤：倾斜帧打地的幻影障碍会堵死 costmap（见 scan_gate.py）
    scan_gate = Node(
        package='tarantula_control',
        executable='scan_gate',
        parameters=[{'use_sim_time': True}],
        output='screen')

    slam = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(bringup_dir, 'launch', 'slam.launch.py')))

    controller = Node(
        package='nav2_controller',
        executable='controller_server',
        parameters=[params_file],
        remappings=[cmd_vel_remap],
        output='screen')

    planner = Node(
        package='nav2_planner',
        executable='planner_server',
        parameters=[params_file],
        output='screen')

    behaviors = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        parameters=[params_file],
        remappings=[cmd_vel_remap],
        output='screen')

    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        parameters=[params_file],
        output='screen')

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        parameters=[{'use_sim_time': True,
                     'autostart': True,
                     'node_names': ['controller_server', 'planner_server',
                                    'behavior_server', 'bt_navigator']}],
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=os.path.join(
            bringup_dir, 'config', 'nav2.yaml')),
        scan_gate,
        slam,
        controller,
        planner,
        behaviors,
        bt_navigator,
        lifecycle_manager,
    ])
