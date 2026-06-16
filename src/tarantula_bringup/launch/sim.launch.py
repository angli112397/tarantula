import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import AndSubstitution, NotSubstitution
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bringup_dir = get_package_share_directory('tarantula_bringup')
    description_dir = get_package_share_directory('tarantula_description')

    gui = LaunchConfiguration('gui')
    world = LaunchConfiguration('world')

    robot_description = ParameterValue(
        Command(['xacro ', os.path.join(description_dir, 'urdf', 'tarantula.urdf.xacro')]),
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
            # 几何接触传感器（诊断用）：<topic> 覆写对 Contact 传感器不生效，
            # gz 侧仍用 scoped 默认名（含 world 名，此处硬编码 rough_terrain）。
            '/world/rough_terrain/model/tarantula/link/wheel_fl_link/sensor/contact_fl/contact'
            '@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            '/world/rough_terrain/model/tarantula/link/wheel_fr_link/sensor/contact_fr/contact'
            '@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            '/world/rough_terrain/model/tarantula/link/wheel_ml_link/sensor/contact_ml/contact'
            '@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            '/world/rough_terrain/model/tarantula/link/wheel_mr_link/sensor/contact_mr/contact'
            '@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            '/world/rough_terrain/model/tarantula/link/wheel_rl_link/sensor/contact_rl/contact'
            '@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            '/world/rough_terrain/model/tarantula/link/wheel_rr_link/sensor/contact_rr/contact'
            '@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
        ],
        remappings=[
            ('/imu', '/imu/data'),
            ('/world/rough_terrain/model/tarantula/link/wheel_fl_link/sensor/contact_fl/contact', '/contact/fl'),
            ('/world/rough_terrain/model/tarantula/link/wheel_fr_link/sensor/contact_fr/contact', '/contact/fr'),
            ('/world/rough_terrain/model/tarantula/link/wheel_ml_link/sensor/contact_ml/contact', '/contact/ml'),
            ('/world/rough_terrain/model/tarantula/link/wheel_mr_link/sensor/contact_mr/contact', '/contact/mr'),
            ('/world/rough_terrain/model/tarantula/link/wheel_rl_link/sensor/contact_rl/contact', '/contact/rl'),
            ('/world/rough_terrain/model/tarantula/link/wheel_rr_link/sensor/contact_rr/contact', '/contact/rr'),
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

    # Classical path: diff_drive_controller (grouped L/R, publishes /odom)
    diff_drive = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
        condition=UnlessCondition(LaunchConfiguration('rl_policy')),
        output='screen')

    # v5 RL path: per-wheel independent velocity controller (replaces diff_drive)
    wheel_velocity_ctrl = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['wheel_velocity_controller', '--controller-manager', '/controller_manager'],
        condition=IfCondition(LaunchConfiguration('rl_policy')),
        output='screen')

    suspension = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['suspension_controller', '--controller-manager', '/controller_manager'],
        output='screen')

    # v4: active_suspension must NOT run when rl_policy:=true (rl_suspension_policy
    # publishes torques directly to /suspension_controller/commands; running both
    # would cause conflicting torque commands on the same topic).
    active_suspension = Node(
        package='tarantula_control',
        executable='active_suspension',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(AndSubstitution(
            LaunchConfiguration('leveling'),
            NotSubstitution(LaunchConfiguration('rl_policy')))),
        output='screen')

    # M7 v4：RL 悬挂+差速驱动策略（PPO actor，src/tarantula_isaac 训练）。
    # v4 直接向 /suspension_controller/commands 发前馈力矩（绕过
    # active_suspension），active_suspension 在 rl_policy:=true 时不启动。
    # move_cmd/heading_rate_cmd 是"方向类"指令接口（停/走、转向意图），
    # 由策略自行决定如何用差速轮实现。
    rl_suspension_policy = Node(
        package='tarantula_control',
        executable='rl_suspension_policy',
        parameters=[{
            'use_sim_time': True,
            'move_cmd': ParameterValue(LaunchConfiguration('move_cmd'), value_type=float),
            'heading_rate_cmd': ParameterValue(LaunchConfiguration('heading_rate_cmd'), value_type=float),
        }],
        condition=IfCondition(LaunchConfiguration('rl_policy')),
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='false 时无图形界面运行（headless）'),
        DeclareLaunchArgument('spawn_x', default_value='0.0'),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_z', default_value='0.38'),
        DeclareLaunchArgument('leveling', default_value='true',
                              description='false 时不启动主动调平（纯被动悬挂对照）'),
        DeclareLaunchArgument('rl_policy', default_value='false',
                              description='true 时启动 M7 v4 RL 策略节点（直接几何映射，绕过 active_suspension）'),
        DeclareLaunchArgument('move_cmd', default_value='1.0',
                              description='RL 策略"方向类"指令：0.0=停, 1.0=走（仅 rl_policy:=true 时生效）'),
        DeclareLaunchArgument('heading_rate_cmd', default_value='0.0',
                              description='RL 策略期望转向角速度 rad/s（仅 rl_policy:=true 时生效）'),
        DeclareLaunchArgument('world', default_value=os.path.join(
            bringup_dir, 'worlds', 'rough_terrain.world')),
        gz_sim_gui,
        gz_sim_headless,
        bridge,
        robot_state_publisher,
        lidar_frame_bridge,
        spawn_robot,
        # 串行启动控制器：spawn 完成 -> jsb -> 两个控制器 -> 避震节点
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_robot, on_exit=[joint_state_broadcaster])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[diff_drive, wheel_velocity_ctrl, suspension])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[active_suspension])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[rl_suspension_policy])),
    ])
