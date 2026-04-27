from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port_arg = DeclareLaunchArgument('port', default_value='8443')
    host_arg = DeclareLaunchArgument('host', default_value='0.0.0.0')
    topic_arg = DeclareLaunchArgument('audio_topic', default_value='/audio_raw')

    return LaunchDescription([
        port_arg, host_arg, topic_arg,
        Node(
            package='go2_phone_mic',
            executable='phone_mic_node',
            name='phone_mic_node',
            output='screen',
            parameters=[{
                'host': LaunchConfiguration('host'),
                'port': LaunchConfiguration('port'),
                'audio_topic': LaunchConfiguration('audio_topic'),
            }],
        ),
    ])
