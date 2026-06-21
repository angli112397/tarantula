import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from tarantula_control.motion_control import MotionControlConfig


def generate_launch_description():
    motion_defaults = MotionControlConfig()
    bringup_dir = get_package_share_directory('tarantula_bringup')
    description_dir = get_package_share_directory('tarantula_description')
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    default_generated_world = os.path.join(repo_root, 'generated', 'terrains', 'nav_maze', '42', 'world.sdf')
    if not os.path.exists(default_generated_world):
        default_generated_world = os.path.abspath(
            os.path.join(os.getcwd(), 'generated', 'terrains', 'nav_maze', '42', 'world.sdf'))
    if not os.path.exists(default_generated_world):
        default_generated_world = os.path.join(repo_root, 'generated', 'terrains', 'gazebo_demo', '42', 'world.sdf')
    if not os.path.exists(default_generated_world):
        default_generated_world = os.path.abspath(
            os.path.join(os.getcwd(), 'generated', 'terrains', 'gazebo_demo', '42', 'world.sdf'))

    gui = LaunchConfiguration('gui')
    world = LaunchConfiguration('world')
    custom_drive = PythonExpression([
        "'", LaunchConfiguration('motion_control'), "' == 'true' and '",
        LaunchConfiguration('drive_controller'), "' == 'custom'"
    ])
    custom_drive_started = PythonExpression([
        "'", LaunchConfiguration('motion_control'), "' == 'true' and '",
        LaunchConfiguration('drive_controller'), "' == 'custom' and '",
        LaunchConfiguration('start_motion_control'), "' == 'true'"
    ])
    diff_drive = PythonExpression([
        "'", LaunchConfiguration('motion_control'), "' == 'true' and '",
        LaunchConfiguration('drive_controller'), "' == 'diff_drive'"
    ])

    robot_description = ParameterValue(
        Command([
            'xacro ',
            os.path.join(description_dir, 'urdf'),
            '/',
            LaunchConfiguration('robot_model'),
            ' wheel_collision:=',
            LaunchConfiguration('wheel_collision'),
        ]),
        value_type=str)

    # gui:=true  -> 带 GUI 运行；gui:=false -> -s 纯 server (headless)
    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r ', world], 'on_exit_shutdown': 'True'}.items(),
        condition=IfCondition(gui))

    gz_sim_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r -s ', world], 'on_exit_shutdown': 'True'}.items(),
        condition=UnlessCondition(gui))

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen')

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'tarantula',
                   '-x', LaunchConfiguration('spawn_x'),
                   '-y', LaunchConfiguration('spawn_y'),
                   '-z', LaunchConfiguration('spawn_z'),
                   '-Y', LaunchConfiguration('spawn_yaw')],
        output='screen')

    # Keep the deployable GUI launch honest: wheel F/T is part of the current
    # RL observation contract. Odometry is inferred from wheel encoders + IMU
    # through robot_localization, not from Gazebo model truth.
    core_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        ],
        remappings=[
            ('/imu', '/imu/data'),
        ],
        output='screen')

    lidar_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        ],
        condition=IfCondition(LaunchConfiguration('bridge_lidar')),
        output='screen')

    force_torque_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/ft/fl@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '/ft/fr@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '/ft/ml@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '/ft/mr@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '/ft/rl@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '/ft/rr@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '/ft_wheel/fl@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '/ft_wheel/fr@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '/ft_wheel/ml@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '/ft_wheel/mr@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '/ft_wheel/rl@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
            '/ft_wheel/rr@geometry_msgs/msg/Wrench[gz.msgs.Wrench',
        ],
        condition=IfCondition(LaunchConfiguration('bridge_force_torque')),
        output='screen')

    # Ground-truth odometry: eval harnesses only (see tarantula_v3.urdf.xacro's
    # OdometryPublisher plugin docstring) -- never bridged into Nav2/control.
    ground_truth_odom_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/ground_truth_odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        ],
        condition=IfCondition(LaunchConfiguration('bridge_ground_truth_odom')),
        output='screen')

    # gz-sim 的 LaserScan.header.frame_id 是 sensor 的 scoped 名（tarantula/base_link/
    # lidar_sensor），与 URDF 的 lidar_link 是同一物理位姿（sensor 在 lidar_link 内无
    # 额外 <pose>）；发一条静态 identity TF 把它接到 lidar_link，否则 slam_toolbox/
    # costmap 的 tf2 MessageFilter 找不到该 frame，扫描会被持续丢弃。
    lidar_frame_bridge = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'lidar_link', 'tarantula/base_link/lidar_sensor'],
        condition=IfCondition(LaunchConfiguration('bridge_lidar')),
        output='screen')

    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen')

    # Custom per-wheel path. The current Nav2 demo baseline uses the official
    # diff_drive_controller instead; this path is retained for chassis
    # commissioning and calibration.
    wheel_velocity_ctrl = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['wheel_velocity_controller', '--controller-manager', '/controller_manager'],
        condition=IfCondition(custom_drive),
        output='screen')

    diff_drive_ctrl = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
        condition=IfCondition(diff_drive),
        output='screen')

    suspension = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['suspension_controller', '--controller-manager', '/controller_manager'],
        output='screen')

    wheel_odometry_node = Node(
        package='tarantula_control',
        executable='wheel_odometry_node',
        parameters=[{
            'use_sim_time': True,
            'effective_track_scale': ParameterValue(LaunchConfiguration('odom_effective_track_scale'), value_type=float),
        }],
        condition=IfCondition(custom_drive),
        output='screen')

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[os.path.join(bringup_dir, 'config', 'ekf.yaml')],
        condition=IfCondition(custom_drive),
        output='screen')

    # Motion deployment node: classical skid-steer wheel targets only.
    # cmd_vx/cmd_wz are cmd_vel-style fallback defaults; /cmd_vel overrides them.
    motion_control_node = Node(
        package='tarantula_control',
        executable='motion_control_node',
        parameters=[{
            'use_sim_time': True,
            'cmd_vx': ParameterValue(LaunchConfiguration('cmd_vx'), value_type=float),
            'cmd_wz': ParameterValue(LaunchConfiguration('cmd_wz'), value_type=float),
            'max_abs_cmd_vx': ParameterValue(LaunchConfiguration('max_abs_cmd_vx'), value_type=float),
            'max_abs_cmd_wz': ParameterValue(LaunchConfiguration('max_abs_cmd_wz'), value_type=float),
            'max_abs_wheel_omega': ParameterValue(LaunchConfiguration('max_abs_wheel_omega'), value_type=float),
            'drive_scale': ParameterValue(LaunchConfiguration('drive_scale'), value_type=float),
            'yaw_track_scale': ParameterValue(LaunchConfiguration('yaw_track_scale'), value_type=float),
            'yaw_rate_kp': ParameterValue(LaunchConfiguration('yaw_rate_kp'), value_type=float),
            'yaw_rate_ki': ParameterValue(LaunchConfiguration('yaw_rate_ki'), value_type=float),
            'yaw_integral_limit': ParameterValue(LaunchConfiguration('yaw_integral_limit'), value_type=float),
            'max_wheel_accel': ParameterValue(LaunchConfiguration('max_wheel_accel'), value_type=float),
        }],
        condition=IfCondition(custom_drive_started),
        output='screen')

    posture_policy_node = Node(
        package='tarantula_control',
        executable='posture_policy_node',
        parameters=[{
            'use_sim_time': True,
            'policy_weights_npz': LaunchConfiguration('policy_weights_npz'),
            'cmd_vx': ParameterValue(LaunchConfiguration('cmd_vx'), value_type=float),
            'cmd_wz': ParameterValue(LaunchConfiguration('cmd_wz'), value_type=float),
            'max_abs_cmd_vx': ParameterValue(LaunchConfiguration('max_abs_cmd_vx'), value_type=float),
            'max_abs_cmd_wz': ParameterValue(LaunchConfiguration('max_abs_cmd_wz'), value_type=float),
            'force_observation_enabled': ParameterValue(LaunchConfiguration('force_observation_enabled'), value_type=bool),
        }],
        condition=IfCondition(LaunchConfiguration('posture_policy_enabled')),
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='false 时无图形界面运行（headless）'),
        # 0.0/0.0 is only guaranteed safe for nav_maze (flat floor, this
        # launch file's actual default world) and gazebo_demo (hand-placed
        # features deliberately kept small/away from its origin). It is NOT
        # safe for rl_curriculum: that grid is centered on (0,0), so world
        # origin sits at a corner where up to 4 curriculum tiles meet, and
        # generate_heightmap no longer flattens it (removed _clear_spawn --
        # smoothstep-blending that junction still left a 20-30deg local
        # slope no radius/feather choice fully fixed). When launching with
        # world:=generated/terrains/rl_curriculum/<seed>/world.sdf, pass
        # spawn_x:=-2.0 spawn_y:=-6.0 explicitly -- that tile's (row=0,
        # col=2) center, difficulty 0, with its own clean _clear_platform
        # square and no neighboring-tile seam.
        DeclareLaunchArgument('spawn_x', default_value='0.0'),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_z', default_value='0.55'),
        DeclareLaunchArgument('spawn_yaw', default_value='0.0'),
        DeclareLaunchArgument('motion_control', default_value='true',
                              description='true 时启动选定的 wheel drive controller'),
        DeclareLaunchArgument('drive_controller', default_value='custom',
                              description='custom 使用 motion_control_node + wheel_velocity_controller；diff_drive 使用官方 diff_drive_controller'),
        DeclareLaunchArgument('start_motion_control', default_value='true',
                              description='drive_controller:=custom 时是否启动 motion_control_node；false 可只启动底层控制器用于开环物理测试'),
        DeclareLaunchArgument('wheel_collision', default_value='sphere',
                              description='轮胎 collision 几何：默认 sphere；cylinder 仅用于接触物理 A/B'),
        DeclareLaunchArgument('robot_model', default_value='tarantula_v3.urdf.xacro',
                              description='robot xacro model under tarantula_description/urdf; current baseline is tarantula_v3.urdf.xacro'),
        DeclareLaunchArgument('cmd_vx', default_value='0.0',
                              description='motion_control 默认前向速度 m/s；/cmd_vel 会覆盖该值'),
        DeclareLaunchArgument('cmd_wz', default_value='0.0',
                              description='motion_control 默认 yaw rate rad/s；/cmd_vel 会覆盖该值'),
        DeclareLaunchArgument('max_abs_cmd_vx', default_value=str(motion_defaults.max_abs_cmd_vx),
                              description='motion_control cmd_vx 限幅 m/s'),
        DeclareLaunchArgument('max_abs_cmd_wz', default_value=str(motion_defaults.max_abs_cmd_wz),
                              description='motion_control cmd_wz 限幅 rad/s'),
        DeclareLaunchArgument('max_abs_wheel_omega', default_value=str(motion_defaults.max_abs_wheel_omega),
                              description='motion_control wheel velocity target clamp rad/s'),
        DeclareLaunchArgument('drive_scale', default_value=str(motion_defaults.drive_scale),
                              description='标定后的传统前进速度有效增益'),
        DeclareLaunchArgument('yaw_track_scale', default_value=str(motion_defaults.yaw_track_scale),
                              description='skid-steer yaw effective track scale；同时作用于曲线和原地 yaw'),
        DeclareLaunchArgument('yaw_rate_kp', default_value=str(motion_defaults.yaw_rate_kp),
                              description='yaw-rate 闭环 P 增益；默认关闭，避免破坏开环几何差速的旋转中心'),
        DeclareLaunchArgument('yaw_rate_ki', default_value=str(motion_defaults.yaw_rate_ki),
                              description='yaw-rate 闭环 I 增益；默认关闭'),
        DeclareLaunchArgument('yaw_integral_limit', default_value=str(motion_defaults.yaw_integral_limit),
                              description='yaw-rate 积分误差限幅 rad'),
        DeclareLaunchArgument('max_wheel_accel', default_value=str(motion_defaults.max_wheel_accel),
                              description='motion_control wheel target slew limit rad/s^2；<=0 关闭'),
        DeclareLaunchArgument('policy_weights_npz', default_value='',
                              description='6D active-suspension RL actor .npz 权重路径；posture_policy_enabled:=true 时必须显式提供'),
        DeclareLaunchArgument('posture_policy_enabled', default_value='false',
                              description='true 时启动 RL 主动悬挂姿态控制；不影响 wheel/cmd_vel 运动链路'),
        DeclareLaunchArgument('force_observation_enabled', default_value='true',
                              description='true 时 posture_policy_node 订阅 /ft_wheel/*；当前 RL observation contract 默认开启'),
        DeclareLaunchArgument('bridge_force_torque', default_value='true',
                              description='true 时桥接 12 路 Gazebo force/torque topic；当前 RL/Gazebo 验收默认开启'),
        DeclareLaunchArgument('bridge_lidar', default_value='false',
                              description='true 时桥接 /scan 并发布 lidar frame 静态 TF；SLAM/Nav 演示时开启'),
        DeclareLaunchArgument('bridge_ground_truth_odom', default_value='false',
                              description='true 时桥接 /ground_truth_odom（仿真真值位姿/速度）；仅供 eval 脚本读取，绝不接入 Nav2/control'),
        DeclareLaunchArgument('odom_effective_track_scale', default_value=str(motion_defaults.yaw_track_scale),
                              description='wheel odom skid-steer effective track scale used before robot_localization fusion'),
        DeclareLaunchArgument('world', default_value=default_generated_world,
                              description='Gazebo world SDF；默认使用 generated/terrains/nav_maze/42/world.sdf'),
        gz_sim_gui,
        gz_sim_headless,
        core_bridge,
        lidar_bridge,
        force_torque_bridge,
        ground_truth_odom_bridge,
        robot_state_publisher,
        lidar_frame_bridge,
        spawn_robot,
        # 串行启动控制器：spawn 完成 -> jsb -> wheel/suspension controllers -> control/diagnostic nodes.
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_robot, on_exit=[joint_state_broadcaster])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[wheel_velocity_ctrl, diff_drive_ctrl, suspension])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[wheel_odometry_node, ekf_node])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[motion_control_node])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[posture_policy_node])),
    ])
