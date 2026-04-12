import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()


    move_service_node = Node(
        package='go2_controller',
        executable='move_service_node',
        name='move_service_node',
        output='screen',
    )

    controller_node = Node(
        package='go2_controller',
        executable='go2_controller_node',
        name='go2_controller_node',
        output='screen',
    )
    
    ld.add_action(move_service_node)
    ld.add_action(controller_node)

    return ld