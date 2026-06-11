import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.conditions import IfCondition
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

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
        launch_arguments={
            'world': world,
            'gui': gui,
        }.items())

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen')

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'tarantula',
                   '-x', LaunchConfiguration('spawn_x'),
                   '-y', LaunchConfiguration('spawn_y'),
                   '-z', LaunchConfiguration('spawn_z')],
        output='screen')

    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen')

    diff_drive = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
        output='screen')

    suspension = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['suspension_controller', '--controller-manager', '/controller_manager'],
        output='screen')

    active_suspension = Node(
        package='tarantula_control',
        executable='active_suspension',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('leveling')),
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='false 时无图形界面运行（headless）'),
        DeclareLaunchArgument('spawn_x', default_value='0.0'),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_z', default_value='0.38'),
        DeclareLaunchArgument('leveling', default_value='true',
                              description='false 时不启动主动调平（纯被动悬挂对照）'),
        DeclareLaunchArgument('world', default_value=os.path.join(
            bringup_dir, 'worlds', 'rough_terrain.world')),
        gazebo,
        robot_state_publisher,
        spawn_robot,
        # 串行启动控制器：spawn 完成 -> jsb -> 两个控制器 -> 避震节点
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_robot, on_exit=[joint_state_broadcaster])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[diff_drive, suspension])),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster, on_exit=[active_suspension])),
    ])
