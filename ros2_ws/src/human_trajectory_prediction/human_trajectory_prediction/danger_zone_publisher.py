import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math
import numpy as np
from my_robot_interfaces.msg import DangerZone
from my_robot_interfaces.msg import DeleteZone

class DangerZonePublisher(Node):
    def __init__(self):
        super().__init__('danger_zone_publisher')
        self.subscription_create = self.create_subscription(
            DangerZone,
            '/add_to_rviz',
            self.listener_callback_add_marker,
            10)
        self.subscription_delete = self.create_subscription(
            DeleteZone,
            '/delete_from_rviz',
            self.listener_callback_delete_marker,
            10)
        self.publisher = self.create_publisher(
            Marker,
            '/visualization_marker',
            10)

    def create_circle(self, ID, CoordX, CoordY, radius, num_points = 100):
        marker = Marker()
        marker.header.frame_id = "body"
        marker.id = ID
        marker.type = marker.CYLINDER
        marker.action = marker.ADD

        self.position = Point(x=CoordX, y=CoordY, z=0)
        self.radius = radius

        marker.pose.position = self.position

        marker.scale.x = self.radius
        marker.scale.y = self.radius
        marker.scale.z = 0.01  # Set to a small value

        marker.color.a = 0.8 # Transparency
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.publisher.publish(marker)
        
        # Alternative with LINE_STRIP
        '''
        marker = Marker()
        marker.header.frame_id = "body"
        marker.id = ID
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        
        self.radius = radius
        
        for i in range(num_points):
            theta = 2.0 * math.pi * i / num_points
            point = Point()
            point.x = radius * math.cos(theta) + CoordX
            point.y = radius * math.sin(theta) + CoordY
            point.z = 0.0
            marker.points.append(point)
            
        marker.scale.x = 0.01  # Set the width of the line

        marker.color.a = 1.0 # Transparency
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.publisher.publish(marker)
        '''
    
    def create_sector(self, ID, CoordX, CoordY, radius, angle, direction, num_points = 100):
        marker = Marker()
        marker.header.frame_id = "body"
        marker.id = ID
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD

        self.radius = radius

        # Create points for the sector
        for i in range(num_points + 1):
            theta = direction + i * angle / num_points
            x = self.radius * math.cos(theta) + CoordX
            y = self.radius * math.sin(theta) + CoordY
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0
            marker.points.append(point)

        marker.scale.x = 0.01  # Set the width of the line

        marker.color.a = 1.0 # Transparency
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.publisher.publish(marker)
    
    def listener_callback_add_marker(self, msg: DangerZone):
        self.get_logger().info(f"log_ADD {msg.id}")
        position = msg.point
        self.create_circle(msg.id,position.x, position.y, msg.r)

        # if msg.theta == 2*np.pi:
        #     self.create_circle(msg.id,position.x, position.y, msg.r)
        # else:
        #     self.create_sector(msg.id,position.x, position.y, msg.r, msg.theta, msg.phi)
         

    def listener_callback_delete_marker(self, msg: DeleteZone): # Call when: human is no longer detected, circle turns into a sector (shape='CYLINDER'), or sector turns into a circle (shape='LINE_STRIP')
        # shape = 'CYLINDER' or 'LINE_STRIP'
        self.get_logger().info(f"log_DELETE {msg.id}")
        shape = msg.shape
        marker = Marker()
        marker.header.frame_id = "body"
        marker.id = msg.id
        if shape == "CYLINDER":
            marker.type = marker.CYLINDER
        else:
            marker.type = marker.LINE_STRIP
        marker.action = marker.DELETE
        self.publisher.publish(marker)

def main(args=None):
        rclpy.init(args=args)
        danger_zone_publisher = DangerZonePublisher()
        rclpy.spin(danger_zone_publisher)
        danger_zone_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
