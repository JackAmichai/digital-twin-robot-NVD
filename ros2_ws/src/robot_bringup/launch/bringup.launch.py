"""
Robot Bringup Launch File
Launches full navigation stack with Nav2
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package paths
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    # Launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file = LaunchConfiguration('params_file', default='/config/nav2_params.yaml')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value='/config/nav2_params.yaml',
            description='Nav2 parameters file'
        ),
        
        # Robot description
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('robot_description'),
                '/launch/robot_description.launch.py'
            ])
        ),
        
        # Nav2 navigation stack
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                nav2_bringup_dir, '/launch/navigation_launch.py'
            ]),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file': params_file,
            }.items()
        ),
        
        # Cognitive bridge
        Node(
            package='cognitive_bridge',
            executable='bridge_node',
            name='cognitive_bridge',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'redis_url': 'redis://redis:6379',
                'redis_channel': 'robot_commands',
            }]
        ),
        
        # Foxglove bridge for visualization
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            parameters=[{
                'use_sim_time': use_sim_time,
                'port': 8765,
            }]
        ),
    ])
