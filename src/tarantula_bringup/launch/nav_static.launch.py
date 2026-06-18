import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Static-map Nav2 bringup for known generated maze maps.

    This launch uses map_server + AMCL instead of online SLAM. It is intended
    for demo navigation on generated maps whose occupancy layer is already
    known and aligned with the Gazebo world.

    Robot spawns at (0, 0, yaw=0) in the navigation baseline by default; pass
    matching initial_pose_x/y/a if spawning elsewhere.
    """
    bringup_dir = get_package_share_directory('tarantula_bringup')
    params_file = LaunchConfiguration('params_file')
    map_file = LaunchConfiguration('map')
    cmd_vel_remap = ('cmd_vel', LaunchConfiguration('cmd_vel_topic'))
    odom_override = {'odom_topic': LaunchConfiguration('odom_topic')}

    scan_gate = Node(
        package='tarantula_control',
        executable='scan_gate',
        parameters=[{'use_sim_time': True}],
        output='screen')

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{
            'use_sim_time': True,
            'yaml_filename': map_file,
        }],
        output='screen')

    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        parameters=[{
            'use_sim_time': True,
            'base_frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'global_frame_id': 'map',
            'scan_topic': '/scan_gated',
            # Provide initial pose matching robot spawn so AMCL converges immediately
            # instead of running global localization across the entire map.
            'set_initial_pose': True,
            'initial_pose.x': ParameterValue(LaunchConfiguration('initial_pose_x'), value_type=float),
            'initial_pose.y': ParameterValue(LaunchConfiguration('initial_pose_y'), value_type=float),
            'initial_pose.yaw': ParameterValue(LaunchConfiguration('initial_pose_a'), value_type=float),
            'transform_tolerance': 0.5,
            'update_min_d': 0.10,
            'update_min_a': 0.10,
            'min_particles': 500,
            'max_particles': 2000,
        }],
        output='screen')

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
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': [
                'map_server',
                'amcl',
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
            ],
        }],
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(bringup_dir, 'config', 'nav2.yaml')),
        DeclareLaunchArgument(
            'map',
            default_value=os.path.join(
                os.getcwd(), 'generated', 'terrains', 'nav_maze', '42', 'map.yaml')),
        DeclareLaunchArgument(
            'cmd_vel_topic', default_value='/cmd_vel',
            description='Nav2 controller output topic; use /diff_drive_controller/cmd_vel_unstamped for official diff_drive_controller'),
        DeclareLaunchArgument(
            'odom_topic', default_value='/odometry/filtered',
            description='Nav2 odometry topic; use /diff_drive_controller/odom for official diff_drive_controller'),
        DeclareLaunchArgument(
            'initial_pose_x', default_value='0.0',
            description='AMCL initial pose X — should match sim.launch.py spawn_x'),
        DeclareLaunchArgument(
            'initial_pose_y', default_value='0.0',
            description='AMCL initial pose Y — should match sim.launch.py spawn_y'),
        DeclareLaunchArgument(
            'initial_pose_a', default_value='0.0',
            description='AMCL initial yaw (rad) — should match robot spawn heading'),
        scan_gate,
        map_server,
        amcl,
        controller,
        planner,
        behaviors,
        bt_navigator,
        lifecycle_manager,
    ])
