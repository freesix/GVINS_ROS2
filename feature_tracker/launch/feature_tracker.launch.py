from launch import LaunchDescription 
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    log_level = 'info'

    share_dir = get_package_share_directory('gvins_feature_tracker')
    # parameter_file = LaunchConfiguration('config_file')
    return LaunchDescription([
        DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(share_dir, 'config/visensor_f9p/visensor_left_f9p_config.yaml'),
        description='Full path to the feature tracker configuration file to load'        
        ),
    
        Node(
            package='gvins_feature_tracker',    
            executable='feature_tracker_node',
            name='my_node',
            parameters=[{'config_file': LaunchConfiguration('config_file')}],
            ros_arguments=['--log-level', log_level]
            # parameters=[{'config_file': 'config/visensor_f9p/visensor_left_f9p_config.yaml'}],
            # output='screen'
        )  
    ])
   

    