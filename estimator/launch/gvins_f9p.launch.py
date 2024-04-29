import os 
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():
    log_level = 'info'

    share_dir = get_package_share_directory('gvins')
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=os.path.join(share_dir, 'config/visensor_f9p/visensor_left_f9p_config.yaml'),
            description='Full path to the feature tracker configuration file to load'        
        ),
        DeclareLaunchArgument(
            'gvins_path',
            default_value='$(find gvins)/../',
            description='gvins path' 
        ),
    
    
        Node(
            package='gvins_feature_tracker',    
            executable='feature_tracker_node',
            name='feature_node',
            parameters=[{'config_file': LaunchConfiguration('config_file')},
                        {'gvins_path': LaunchConfiguration('gvins_path')}],
            ros_arguments=['--log-level', log_level],
            output='screen'
        ),
        Node(
            package='gvins',
            executable='estimator_node',
            name='estimator_node',
            parameters=[{'config_file': LaunchConfiguration('config_file')},
                        {'gvins_path': LaunchConfiguration('gvins_path')}],
            ros_arguments=['--log-level', log_level],
            output='screen'
        )
    ])
