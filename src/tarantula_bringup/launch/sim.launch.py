import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import AndSubstitution, NotSubstitution, OrSubstitution
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
            os.path.join(description_dir, 'urdf', 'tarantula.urdf.xacro'),
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

    per_wheel_mode = OrSubstitution(
        LaunchConfiguration('rl_policy'),
        LaunchConfiguration('manual_wheel'))

    # Classical path: diff_drive_controller (grouped L/R, publishes /odom)
    diff_drive = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
        condition=UnlessCondition(per_wheel_mode),
        output='screen')

    # Per-wheel path: manual_wheel baseline or RL, both replace diff_drive.
    wheel_velocity_ctrl = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['wheel_velocity_controller', '--controller-manager', '/controller_manager'],
        condition=IfCondition(per_wheel_mode),
        output='screen')

    suspension = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['suspension_controller', '--controller-manager', '/controller_manager'],
        output='screen')

    # RL path: active_suspension must NOT run when rl_policy:=true (rl_suspension_policy
    # publishes torques directly to /suspension_controller/commands; running both
    # would cause conflicting torque commands on the same topic).
    active_suspension = Node(
        package='tarantula_control',
        executable='active_suspension',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(AndSubstitution(
            LaunchConfiguration('leveling'),
            AndSubstitution(
                NotSubstitution(LaunchConfiguration('stand_hold')),
                OrSubstitution(
                    NotSubstitution(LaunchConfiguration('rl_policy')),
                    LaunchConfiguration('start_active_suspension'))))),
        output='screen')

    stand_suspension_hold = Node(
        package='tarantula_control',
        executable='stand_suspension_hold',
        parameters=[{
            'use_sim_time': True,
            'target': ParameterValue(LaunchConfiguration('stand_target'), value_type=float),
            'kp': ParameterValue(LaunchConfiguration('stand_kp'), value_type=float),
            'kd': ParameterValue(LaunchConfiguration('stand_kd'), value_type=float),
            'effort_limit': ParameterValue(LaunchConfiguration('stand_effort_limit'), value_type=float),
            'target_ramp_rate': ParameterValue(LaunchConfiguration('stand_ramp_rate'), value_type=float),
        }],
        condition=IfCondition(LaunchConfiguration('stand_hold')),
        output='screen')

    cmd_vel_wheel_baseline = Node(
        package='tarantula_control',
        executable='cmd_vel_wheel_baseline',
        parameters=[{
            'use_sim_time': True,
            'cmd_vx': ParameterValue(LaunchConfiguration('cmd_vx'), value_type=float),
            'cmd_wz': ParameterValue(LaunchConfiguration('cmd_wz'), value_type=float),
            'max_abs_cmd_vx': ParameterValue(LaunchConfiguration('max_abs_cmd_vx'), value_type=float),
            'max_abs_cmd_wz': ParameterValue(LaunchConfiguration('max_abs_cmd_wz'), value_type=float),
            'max_abs_wheel_omega': ParameterValue(LaunchConfiguration('max_abs_wheel_omega'), value_type=float),
        }],
        condition=IfCondition(AndSubstitution(
            LaunchConfiguration('manual_wheel'),
            NotSubstitution(LaunchConfiguration('rl_policy')))),
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
            LaunchConfiguration('rl_policy'),
            LaunchConfiguration('truth_odom'))),
        output='screen')

    # M7 v5：RL 悬挂+独立轮速策略（PPO actor，src/tarantula_isaac 训练）。
    # v5 直接向 /suspension_controller/commands 发关节力矩，并向
    # /wheel_velocity_controller/commands 发 6 路轮速（绕过
    # active_suspension），active_suspension 在 rl_policy:=true 时不启动。
    # cmd_vx/cmd_wz 是 cmd_vel-style 指令接口；节点同时订阅 /cmd_vel，
    # launch 参数只作为没有上层导航/遥控输入时的默认指令。
    rl_suspension_policy = Node(
        package='tarantula_control',
        executable='rl_suspension_policy',
        parameters=[{
            'use_sim_time': True,
            'cmd_vx': ParameterValue(LaunchConfiguration('cmd_vx'), value_type=float),
            'cmd_wz': ParameterValue(LaunchConfiguration('cmd_wz'), value_type=float),
            'max_abs_cmd_vx': ParameterValue(LaunchConfiguration('max_abs_cmd_vx'), value_type=float),
            'max_abs_cmd_wz': ParameterValue(LaunchConfiguration('max_abs_cmd_wz'), value_type=float),
            'policy_weights_npz': LaunchConfiguration('policy_weights_npz'),
            'policy_mode': LaunchConfiguration('rl_policy_mode'),
            'velocity_source': LaunchConfiguration('velocity_source'),
            'truth_odom_topic': LaunchConfiguration('truth_odom_topic'),
        }],
        condition=IfCondition(AndSubstitution(
            LaunchConfiguration('rl_policy'),
            LaunchConfiguration('start_rl_policy'))),
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
                              description='true 时启动 M7 v5 RL 策略节点（每关节力矩 + 每轮速度）'),
        DeclareLaunchArgument('manual_wheel', default_value='false',
                              description='true 时启动 /cmd_vel -> per-wheel 轮速 baseline，使用与 RL 相同的 wheel_velocity_controller'),
        DeclareLaunchArgument('start_rl_policy', default_value='true',
                              description='rl_policy:=true 时是否启动 RL 节点；false 可只启动独立轮速/悬挂控制器用于开环物理测试'),
        DeclareLaunchArgument('start_active_suspension', default_value='false',
                              description='rl_policy:=true 时允许启动 active_suspension；用于轮速开环、悬挂经典调平的物理测试'),
        DeclareLaunchArgument('stand_hold', default_value='false',
                              description='true 时启动独立站姿保持节点，直接 PD 保持悬挂关节目标；与 active_suspension/RL 悬挂输出互斥使用'),
        DeclareLaunchArgument('stand_target', default_value='0.0',
                              description='stand_hold 悬挂关节目标 rad'),
        DeclareLaunchArgument('stand_kp', default_value='95.0',
                              description='stand_hold 悬挂关节 PD kp Nm/rad'),
        DeclareLaunchArgument('stand_kd', default_value='18.0',
                              description='stand_hold 悬挂关节 PD kd Nms/rad'),
        DeclareLaunchArgument('stand_effort_limit', default_value='45.0',
                              description='stand_hold 单关节力矩限幅 Nm'),
        DeclareLaunchArgument('stand_ramp_rate', default_value='0.18',
                              description='stand_hold 目标角 ramp rate rad/s'),
        DeclareLaunchArgument('wheel_collision', default_value='sphere',
                              description='轮胎 collision 几何：sphere 或 cylinder；用于 Gazebo/Isaac 物理 A/B'),
        DeclareLaunchArgument('cmd_vx', default_value='0.2',
                              description='RL/manual_wheel 默认前向速度 m/s；/cmd_vel 会覆盖该值'),
        DeclareLaunchArgument('cmd_wz', default_value='0.0',
                              description='RL/manual_wheel 默认 yaw rate rad/s；/cmd_vel 会覆盖该值'),
        DeclareLaunchArgument('max_abs_cmd_vx', default_value='0.3',
                              description='RL/manual_wheel cmd_vx 限幅 m/s'),
        DeclareLaunchArgument('max_abs_cmd_wz', default_value='0.4',
                              description='RL/manual_wheel cmd_wz 限幅 rad/s'),
        DeclareLaunchArgument('max_abs_wheel_omega', default_value='3.0',
                              description='manual_wheel 单轮速度限幅 rad/s'),
        DeclareLaunchArgument('policy_weights_npz', default_value='',
                              description='RL actor .npz 权重路径；rl_policy:=true 时必须显式提供'),
        DeclareLaunchArgument('rl_policy_mode', default_value='auto',
                              description='auto/wheel_only/suspension_wheel；Stage A 使用 wheel_only 防止与 stand_hold 冲突'),
        DeclareLaunchArgument('velocity_source', default_value='auto',
                              description='RL 速度观测来源：auto/truth_odom/wheel；Gazebo 验证默认 auto 使用 truth_odom'),
        DeclareLaunchArgument('truth_odom', default_value='false',
                              description='rl_policy:=true 时可启动 Gazebo truth odometry 诊断适配节点；默认关闭以避免 CLI 采样拖慢 GUI'),
        DeclareLaunchArgument('truth_odom_topic', default_value='/tarantula/truth_odom',
                              description='Gazebo truth odometry topic for RL deployment observation'),
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
        # 串行启动控制器：spawn 完成 -> jsb -> 两个控制器 -> 避震节点
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_robot, on_exit=[joint_state_broadcaster])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[diff_drive, wheel_velocity_ctrl, suspension])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[active_suspension])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[stand_suspension_hold])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[cmd_vel_wheel_baseline])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[gazebo_truth_odometry])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[rl_suspension_policy])),
    ])
