import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import AndSubstitution
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    description_dir = get_package_share_directory('tarantula_description')
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    default_generated_world = os.path.join(repo_root, 'generated', 'terrains', 'gazebo_demo', '42', 'world.sdf')
    if not os.path.exists(default_generated_world):
        default_generated_world = os.path.abspath(
            os.path.join(os.getcwd(), 'generated', 'terrains', 'gazebo_demo', '42', 'world.sdf'))

    gui = LaunchConfiguration('gui')
    world = LaunchConfiguration('world')

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
                   '-z', LaunchConfiguration('spawn_z')],
        output='screen')

    # gz -> ROS 单向桥接：/clock (use_sim_time)、IMU、雷达、6 路腿部力/力矩传感器
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
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
        remappings=[
            ('/imu', '/imu/data'),
        ],
        output='screen')

    # gz-sim 的 LaserScan.header.frame_id 是 sensor 的 scoped 名（tarantula/base_link/
    # lidar_sensor），与 URDF 的 lidar_link 是同一物理位姿（sensor 在 lidar_link 内无
    # 额外 <pose>）；发一条静态 identity TF 把它接到 lidar_link，否则 slam_toolbox/
    # costmap 的 tf2 MessageFilter 找不到该 frame，扫描会被持续丢弃。
    lidar_frame_bridge = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'lidar_link', 'tarantula/base_link/lidar_sensor'],
        output='screen')

    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen')

    # Per-wheel path: classical motion control with optional RL compensation.
    wheel_velocity_ctrl = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['wheel_velocity_controller', '--controller-manager', '/controller_manager'],
        condition=IfCondition(LaunchConfiguration('motion_control')),
        output='screen')

    suspension = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['suspension_controller', '--controller-manager', '/controller_manager'],
        output='screen')

    gazebo_truth_odometry = Node(
        package='tarantula_control',
        executable='gazebo_truth_odometry',
        parameters=[{
            'use_sim_time': True,
            'model_name': 'tarantula',
            'topic': LaunchConfiguration('truth_odom_topic'),
            'rate': ParameterValue(LaunchConfiguration('truth_odom_rate'), value_type=float),
        }],
        condition=IfCondition(AndSubstitution(
            LaunchConfiguration('motion_control'),
            LaunchConfiguration('truth_odom'))),
        output='screen')

    # Motion deployment node: classical skid-steer wheel targets with optional
    # bounded RL structured compensation. Stage A does not command hip posture;
    # v2 posture is owned by the trajectory controller or an explicit test/profile node.
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
            'arc_track_scale': ParameterValue(LaunchConfiguration('arc_track_scale'), value_type=float),
            'pure_turn_track_scale': ParameterValue(LaunchConfiguration('pure_turn_track_scale'), value_type=float),
            'track_scale_transition_vx': ParameterValue(LaunchConfiguration('track_scale_transition_vx'), value_type=float),
            'track_scale_delta_limit': ParameterValue(LaunchConfiguration('track_scale_delta_limit'), value_type=float),
            'drive_scale_delta_limit': ParameterValue(LaunchConfiguration('drive_scale_delta_limit'), value_type=float),
            'yaw_rate_kp': ParameterValue(LaunchConfiguration('yaw_rate_kp'), value_type=float),
            'yaw_rate_ki': ParameterValue(LaunchConfiguration('yaw_rate_ki'), value_type=float),
            'yaw_integral_limit': ParameterValue(LaunchConfiguration('yaw_integral_limit'), value_type=float),
            'max_wheel_accel': ParameterValue(LaunchConfiguration('max_wheel_accel'), value_type=float),
            'pure_turn_forward_bias': ParameterValue(LaunchConfiguration('pure_turn_forward_bias'), value_type=float),
            'pure_turn_vx_deadband': ParameterValue(LaunchConfiguration('pure_turn_vx_deadband'), value_type=float),
            'turn_enter_wz': ParameterValue(LaunchConfiguration('turn_enter_wz'), value_type=float),
            'turn_exit_wz': ParameterValue(LaunchConfiguration('turn_exit_wz'), value_type=float),
            'command_strategy': LaunchConfiguration('command_strategy'),
            'policy_weights_npz': LaunchConfiguration('policy_weights_npz'),
            'rl_compensation_enabled': ParameterValue(LaunchConfiguration('rl_compensation_enabled'), value_type=bool),
        }],
        condition=IfCondition(AndSubstitution(
            LaunchConfiguration('motion_control'),
            LaunchConfiguration('start_motion_control'))),
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='false 时无图形界面运行（headless）'),
        DeclareLaunchArgument('spawn_x', default_value='0.0'),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_z', default_value='0.55'),
        DeclareLaunchArgument('motion_control', default_value='true',
                              description='true 时启动 /cmd_vel -> per-wheel 传统主控，可选叠加 RL structured compensation'),
        DeclareLaunchArgument('start_motion_control', default_value='true',
                              description='motion_control:=true 时是否启动 motion_control_node；false 可只启动独立轮速/悬挂控制器用于开环物理测试'),
        DeclareLaunchArgument('wheel_collision', default_value='cylinder',
                              description='轮胎 collision 几何：sphere 或 cylinder；用于 Gazebo/Isaac 物理 A/B'),
        DeclareLaunchArgument('robot_model', default_value='tarantula_v2.urdf.xacro',
                              description='robot xacro model under tarantula_description/urdf; current baseline is tarantula_v2.urdf.xacro'),
        DeclareLaunchArgument('cmd_vx', default_value='0.1',
                              description='motion_control 默认前向速度 m/s；/cmd_vel 会覆盖该值'),
        DeclareLaunchArgument('cmd_wz', default_value='0.0',
                              description='motion_control 默认 yaw rate rad/s；/cmd_vel 会覆盖该值'),
        DeclareLaunchArgument('max_abs_cmd_vx', default_value='0.3',
                              description='motion_control cmd_vx 限幅 m/s'),
        DeclareLaunchArgument('max_abs_cmd_wz', default_value='0.4',
                              description='motion_control cmd_wz 限幅 rad/s'),
        DeclareLaunchArgument('arc_track_scale', default_value='1.0',
                              description='continuous A/B 模式下行进弧线的 skid-steer effective track scale'),
        DeclareLaunchArgument('pure_turn_track_scale', default_value='3.0',
                              description='纯转向/低线速度时的 skid-steer effective track scale'),
        DeclareLaunchArgument('track_scale_transition_vx', default_value='0.08',
                              description='continuous A/B 模式下 track scale 从纯转向向弧线行驶过渡的 |cmd_vx| 阈值 m/s'),
        DeclareLaunchArgument('track_scale_delta_limit', default_value='0.3',
                              description='RL 可调 effective track scale 的最大比例补偿'),
        DeclareLaunchArgument('drive_scale_delta_limit', default_value='0.2',
                              description='RL 可调 left/right drive scale 的最大比例补偿'),
        DeclareLaunchArgument('yaw_rate_kp', default_value='0.0',
                              description='yaw-rate 闭环 P 增益；默认关闭，避免破坏开环几何差速的旋转中心'),
        DeclareLaunchArgument('yaw_rate_ki', default_value='0.0',
                              description='yaw-rate 闭环 I 增益；默认关闭'),
        DeclareLaunchArgument('yaw_integral_limit', default_value='0.8',
                              description='yaw-rate 积分误差限幅 rad'),
        DeclareLaunchArgument('max_wheel_accel', default_value='12.0',
                              description='motion_control wheel target slew limit rad/s^2；<=0 关闭'),
        DeclareLaunchArgument('pure_turn_forward_bias', default_value='0.0',
                              description='纯转向时用于抵消 Gazebo 接触倒车漂移的最大前向补偿 m/s'),
        DeclareLaunchArgument('pure_turn_vx_deadband', default_value='0.03',
                              description='|cmd_vx| 小于该值时才启用 pure_turn_forward_bias'),
        DeclareLaunchArgument('turn_enter_wz', default_value='0.08',
                              description='stop_turn_drive: |cmd_wz| 达到该值时执行纯转向，并把执行 cmd_vx 置 0'),
        DeclareLaunchArgument('turn_exit_wz', default_value='0.04',
                              description='stop_turn_drive: |cmd_wz| 低于该值时退出纯转向，回到直行/停止'),
        DeclareLaunchArgument('command_strategy', default_value='stop_turn_drive',
                              description='motion command shaping: stop_turn_drive 或 continuous'),
        DeclareLaunchArgument('policy_weights_npz', default_value='',
                              description='RL actor .npz 权重路径；rl_compensation_enabled:=true 时必须显式提供'),
        DeclareLaunchArgument('rl_compensation_enabled', default_value='false',
                              description='true 时在传统 skid-steer wheel target 上叠加 RL structured compensation；false 时只跑传统运动控制'),
        DeclareLaunchArgument('truth_odom', default_value='false',
                              description='仅启动 Gazebo truth odometry 诊断/benchmark 节点；不进入 motion_control 算法'),
        DeclareLaunchArgument('truth_odom_topic', default_value='/tarantula/truth_odom',
                              description='Gazebo truth odometry 诊断 topic'),
        DeclareLaunchArgument('truth_odom_rate', default_value='5.0',
                              description='Gazebo truth odometry publish rate Hz'),
        DeclareLaunchArgument('world', default_value=default_generated_world,
                              description='Gazebo world SDF；默认使用 generated/terrains/gazebo_demo/42/world.sdf'),
        gz_sim_gui,
        gz_sim_headless,
        bridge,
        robot_state_publisher,
        lidar_frame_bridge,
        spawn_robot,
        # 串行启动控制器：spawn 完成 -> jsb -> wheel/suspension controllers -> control/diagnostic nodes.
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_robot, on_exit=[joint_state_broadcaster])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[wheel_velocity_ctrl, suspension])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[gazebo_truth_odometry])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[motion_control_node])),
    ])
