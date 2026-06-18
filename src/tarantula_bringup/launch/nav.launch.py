import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """SLAM 在线建图 + Nav2 导航（先 ros2 launch tarantula_bringup sim.launch.py）。

    自建最小 Nav2 bringup 而非 include nav2_bringup。map->odom 由
    slam_toolbox 提供，无 AMCL/map_server；cmd_vel 和 odom topic 通过
    launch 参数选择，便于接官方 diff_drive_controller 或项目自定义主控。
    """
    bringup_dir = get_package_share_directory('tarantula_bringup')
    params_file = LaunchConfiguration('params_file')
    cmd_vel_remap = ('cmd_vel', LaunchConfiguration('cmd_vel_topic'))
    odom_override = {'odom_topic': LaunchConfiguration('odom_topic')}

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
        parameters=[params_file, odom_override],
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
        parameters=[params_file, odom_override],
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
        DeclareLaunchArgument(
            'cmd_vel_topic', default_value='/cmd_vel',
            description='Nav2 controller output topic; use /diff_drive_controller/cmd_vel_unstamped for official diff_drive_controller'),
        DeclareLaunchArgument(
            'odom_topic', default_value='/odometry/filtered',
            description='Nav2 odometry topic; use /diff_drive_controller/odom for official diff_drive_controller'),
        scan_gate,
        slam,
        controller,
        planner,
        behaviors,
        bt_navigator,
        lifecycle_manager,
    ])
