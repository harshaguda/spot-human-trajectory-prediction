import cv2
import numpy as np
import rclpy
from rclpy.node import Node

class LidarImage(Node):
    def __init__(self):
        super().__init__('lidar_image')
        self.subscription
        self.angleFactor = 360/(901*5)

    def get_angle(self, mean_index):
        return mean_index*self.angleFactor
    def get_distance(self, mean_index):
        return self.im[mean_index]
