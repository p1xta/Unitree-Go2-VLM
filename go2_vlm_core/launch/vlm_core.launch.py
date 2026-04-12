import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='go2_vlm_core',
            executable='vlm_core_node',
            name='vlm_core_node',
            output='screen',
        ),
    ])
