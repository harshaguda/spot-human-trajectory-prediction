import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ListenerNode(Node):

    def __init__(self):
        super().__init__('listener_node')
        self.subscription = self.create_subscription(
            String,
            '/saved_information_topic',  # Replace with the name of the topic published by PredictionNode
            self.listener_callback,
            10)

    def listener_callback(self, msg: String):
        self.get_logger().info(f'Received data: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = ListenerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
