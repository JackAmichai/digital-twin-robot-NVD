"""
Multi-Robot Fleet Launch Configuration

This launch file starts a fleet of robots with:
- Unique namespaces for each robot (/robot_1, /robot_2, etc.)
- Individual Nav2 stacks per robot
- Shared fleet manager for coordination

Usage:
    # Launch 3-robot fleet
    ros2 launch fleet_management multi_robot_launch.py num_robots:=3
    
    # Launch with custom robot names
    ros2 launch fleet_management multi_robot_launch.py robot_names:="['amr_001','amr_002']"
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_robot_group(context, robot_id: str, namespace: str, 
                         x: float, y: float, yaw: float):
    """
    Generate launch group for a single robot
    
    Each robot gets:
    - Unique namespace (e.g., /robot_1)
    - Its own Nav2 stack
    - Robot agent for fleet communication
    - Robot state publisher
    """
    
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    return GroupAction([
        # Push all nodes into robot's namespace
        PushRosNamespace(namespace),
        
        # Robot Agent (fleet communication)
        Node(
            package='fleet_management',
            executable='robot_agent',
            name='robot_agent',
            parameters=[{
                'robot_id': robot_id,
                'priority': 1,
                'capabilities': ['navigation'],
            }],
            output='screen',
            remappings=[
                # Remap to namespaced topics
                ('odom', 'odom'),
                ('cmd_vel', 'cmd_vel'),
                ('battery_state', 'battery_state'),
            ]
        ),
        
        # Nav2 Stack for this robot
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
            ),
            launch_arguments={
                'namespace': namespace,
                'use_sim_time': 'true',
                'autostart': 'true',
                'params_file': os.path.join(
                    get_package_share_directory('digital_twin_robot'),
                    'config', 'nav2_params.yaml'
                ),
                'use_lifecycle_mgr': 'false',
            }.items()
        ),
        
        # Initial pose publisher (spawn position)
        Node(
            package='robot_localization',
            executable='set_pose',
            name='set_initial_pose',
            output='screen',
            arguments=[
                '--frame', 'map',
                '--x', str(x),
                '--y', str(y),
                '--yaw', str(yaw),
            ],
        ),
    ])


def launch_setup(context, *args, **kwargs):
    """
    Dynamic launch setup based on number of robots
    
    Robot spawn positions (warehouse layout):
    
        Robot 1 (0,0)    Robot 2 (0,3)    Robot 3 (0,6)
             │                │                │
             └────────────────┴────────────────┘
                        Charging Area
    """
    
    num_robots = int(LaunchConfiguration('num_robots').perform(context))
    
    # Default spawn positions (staggered at charging area)
    spawn_positions = [
        (0.0, 0.0, 0.0),    # Robot 1
        (0.0, 3.0, 0.0),    # Robot 2
        (0.0, 6.0, 0.0),    # Robot 3
        (0.0, 9.0, 0.0),    # Robot 4
        (3.0, 0.0, 0.0),    # Robot 5
        (3.0, 3.0, 0.0),    # Robot 6
    ]
    
    launch_items = []
    
    for i in range(min(num_robots, len(spawn_positions))):
        robot_id = f'amr_{i+1:03d}'
        namespace = f'/robot_{i+1}'
        x, y, yaw = spawn_positions[i]
        
        launch_items.append(
            generate_robot_group(context, robot_id, namespace, x, y, yaw)
        )
    
    return launch_items


def generate_launch_description():
    """Generate launch description for multi-robot fleet"""
    
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument(
            'num_robots',
            default_value='3',
            description='Number of robots to spawn'
        ),
        
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        
        # Fleet Manager (central coordinator) - runs in global namespace
        Node(
            package='fleet_management',
            executable='fleet_manager',
            name='fleet_manager',
            output='screen',
            parameters=[{
                'collision_distance': 1.5,
                'warning_distance': 3.0,
                'heartbeat_timeout': 5.0,
            }]
        ),
        
        # Dynamic robot spawning
        OpaqueFunction(function=launch_setup),
    ])
