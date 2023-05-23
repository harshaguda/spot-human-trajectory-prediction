from launch import LaunchDescription
from launch_ros.actions import Node
# from launch.actions import TimerAction


def generate_launch_description():
    ld = LaunchDescription()

    danger_zone_publisher = Node(
        package="human_trajectory_prediction",
        executable="danger_zone_publisher",
        name="danger_zone_publisher",
    )

    prediction_node = Node(
        package="human_trajectory_prediction",
        executable="prediction_node",
        name="prediction_node",
    )

    # Add a delay of 0.1 seconds between launching the nodes
    # delay_action = TimerAction(period=0.1)

    
    # ld.add_action(delay_action)
    ld.add_action(prediction_node)
    ld.add_action(danger_zone_publisher)

    return ld
