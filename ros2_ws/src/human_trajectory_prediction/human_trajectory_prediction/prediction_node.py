from rclpy.node import Node
import rclpy
from std_msgs.msg import String
from typing import Dict, List, Tuple
import numpy as np
from geometry_msgs.msg import Point
import numpy as np
from my_robot_interfaces.msg import DangerZone
from my_robot_interfaces.msg import DeleteZone
import time
import json

class KalmanFilter:
    def __init__(self, x0, P0, r, dt, q):
        """
        Initializes the Kalman Filter.

        Args:
            x (numpy array): The initial state vector.
            P (numpy array): The initial error covariance matrix.
            F (numpy array): The state transition matrix.
            Q (numpy array): The process noise covariance matrix.
            H (numpy array): The measurement function matrix.
            R (numpy array): The measurement noise covariance matrix.
        """

        self.x = x0
        self.dt = dt
        self.P = P0
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.Q = np.array([[dt**3/3, 0, dt**2/2, 0],
                           [0, dt**3/3, 0, dt**2/2],
                           [dt**2/2, 0, dt, 0],
                           [0, dt**2/2, 0, dt]]) * q**2
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.R = r ** 2 * np.eye(2)

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x0 = self.x
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(len(self.x)) - np.dot(K, self.H)), self.P)


class Person:

    def __init__(self, x0, P0, r, dt, q):
        self.ekf = KalmanFilter(x0, P0, r, dt, q)
        # self.danger_zone_publisher = DangerZonePublisher(x0)
        self.r_static = 0.55
        self.mv = 1.3
        self.danger_zone = (self.r_static, 2*np.pi, 0)

    def updateInfo(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.ekf.predict()
        self.ekf.update(measured)
        phi = np.arctan2(self.ekf.x[3], self.ekf.x[2])
        v = self.ekf.x[2] * np.cos(phi) + self.ekf.x[3] * np.sin(phi)
        r = self.mv * v + self.r_static
        if v == 0:
            theta = 2*np.pi
        else:
            theta = (11*np.pi/6) * np.exp(-1.4*v) + np.pi/6
        self.danger_zone = (r, theta, phi)


class PredictionNode(Node):

    def __init__(self):
        super().__init__('prediction_node')
        self.subscription = self.create_subscription(
            String,
            '/trajectories',  # Includes ID, distance to human and angle to human
            self.listener_callback,
            10)
        self.publisher_add = self.create_publisher(
            DangerZone,
            '/add_to_rviz',
            10)
        self.publisher_delete = self.create_publisher(
            DeleteZone,
            '/delete_from_rviz',
            10)
        self.saved_data = {}
        self.dt = 0.01  # update that
        self.r = 0.5  # update that
        self.q = 0.1  # update that

    def angle_xy(self, r, theta):
        # calculates the x- and y-coordinates from radius (r) and angle (theta)
        return r*np.cos(theta), r*np.sin(theta)

    def listener_callback(self, msg: String):
        # convert message to dictionary
        data = json.loads(msg.data)
        output_dict = {key: (value['distance'], value['angle']) for key, value in data.items()}
        data = output_dict
        if not isinstance(data, dict):
            self.get_logger().warn('Received invalid message data type')
            return
        # remove any IDs from saved data that are not in the received message
        for ID in list(self.saved_data.keys()):
            if ID not in data:
                msg = DeleteZone()
                msg.id = int(ID)
                person = self.saved_data[ID]
                if person.danger_zone[1] == 2*np.pi:
                    msg.shape = "CYLINDER"
                else:
                    msg.shape = "LINE_STRIP"
                self.publisher_delete.publish(msg)
                time.sleep(0.01)
                self.get_logger().info(f"delete {ID}")
                del self.saved_data[ID]
    
        # process dictionary
        for ID, coord in data.items():
            if not isinstance(coord, tuple):
                self.get_logger().warn(
                    f'Received invalid coordinates for ID {ID}')
                continue
            r, theta = coord
            x, y = self.angle_xy(r, theta)
            # add data to saved dictionary as a kalman filter object
            if ID not in self.saved_data:
                P0 = np.eye(4)
                x0 = np.array([[x], [y], [0], [0]], dtype=np.float32)
                self.saved_data[ID] = Person(
                    x0, P0, self.r, self.dt, self.q)
                position = Point(
                    x=float(x0[0]), y=float(x0[1]), z=float(0))
                msg = DangerZone()
                msg.point = position
                msg.id = int(ID)
                msg.r = 0.5
                msg.theta = 2*np.pi
                self.publisher_add.publish(msg)
                time.sleep(0.01)
                self.get_logger().info(f"Add first {ID}")
            else:
                person = self.saved_data[ID]
                person.updateInfo(x, y)
                x_updated = person.ekf.x
                position = Point(x=float(x_updated[0]), y=float(
                    x_updated[1]), z=float(0))
                msg = DangerZone()
                msg.point = position
                msg.id = int(ID)
                msg.r = float(person.danger_zone[0])
                msg.theta = float(person.danger_zone[1])
                msg.phi = float(person.danger_zone[2])
                self.publisher_add.publish(msg)
                time.sleep(0.01)
                self.get_logger().info(f"Add second {ID}")


def main(args=None):
    rclpy.init(args=args)
    node = PredictionNode()
    # Simulate receiving messages
    messages = [
        '{"1": {"distance": 4, "angle": 2.14}}'
    ]
    for msg in messages:
        string_msg = String()
        string_msg.data = msg
        node.listener_callback(string_msg)

    # node.get_logger().info('Finished processing messages')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
