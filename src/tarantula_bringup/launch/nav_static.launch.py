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
    localization_map_file = LaunchConfiguration('localization_map')
    terrain_cost_map_file = LaunchConfiguration('terrain_cost_map')
    speed_mask_file = LaunchConfiguration('speed_mask')
    cmd_vel_remap = ('cmd_vel', LaunchConfiguration('cmd_vel_topic'))
    odom_override = {'odom_topic': LaunchConfiguration('odom_topic')}

    # Static-map localization needs continuous scan updates. On the
    # mesh-contact validation world, scan_gate.py's 3 deg default starves AMCL
    # as soon as the robot climbs a small bump, freezing map->odom. One shared
    # 8 deg gate works for both worlds we ship: the flat-floor baseline barely
    # tilts at all (passive/frozen suspension, no terrain relief), so it never
    # gets near either threshold — loosening the gate costs nothing there
    # while fixing the mesh-contact case. No per-world toggle needed.
    scan_gate = Node(
        package='tarantula_control',
        executable='scan_gate',
        parameters=[{
            'use_sim_time': True,
            'tilt_gate': 0.14,
        }],
        output='screen')

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{
            'use_sim_time': True,
            'yaml_filename': localization_map_file,
            'topic_name': 'map',
        }],
        output='screen')

    terrain_map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='terrain_map_server',
        parameters=[{
            'use_sim_time': True,
            'yaml_filename': terrain_cost_map_file,
            'topic_name': 'terrain_cost_map',
        }],
        output='screen')

    speed_mask_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='speed_mask_server',
        parameters=[{
            'use_sim_time': True,
            'yaml_filename': speed_mask_file,
            'topic_name': 'terrain_speed_mask',
        }],
        output='screen')

    speed_filter_info_server = Node(
        package='nav2_map_server',
        executable='costmap_filter_info_server',
        name='speed_costmap_filter_info_server',
        parameters=[{
            'use_sim_time': True,
            'type': 1,
            'filter_info_topic': '/speed_costmap_filter_info',
            'mask_topic': '/terrain_speed_mask',
            'base': 100.0,
            'multiplier': -1.0,
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
            'transform_tolerance': 1.5,
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
                'terrain_map_server',
                'speed_mask_server',
                'speed_costmap_filter_info_server',
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
            'localization_map',
            default_value=os.path.join(
                os.getcwd(), 'generated', 'terrains', 'nav_maze', '42', 'map.yaml'),
            description='Pure occupancy map for AMCL localization'),
        DeclareLaunchArgument(
            'terrain_cost_map',
            default_value=os.path.join(
                os.getcwd(), 'generated', 'terrains', 'nav_maze', '42', 'terrain_cost_map.yaml'),
            description='Terrain-aware scaled cost map for Nav2 global/local costmaps'),
        DeclareLaunchArgument(
            'speed_mask',
            default_value=os.path.join(
                os.getcwd(), 'generated', 'terrains', 'nav_maze', '42', 'terrain_speed_mask.yaml'),
            description='Terrain speed-filter mask for Nav2 SpeedFilter'),
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
        terrain_map_server,
        speed_mask_server,
        speed_filter_info_server,
        amcl,
        controller,
        planner,
        behaviors,
        bt_navigator,
        lifecycle_manager,
    ])
