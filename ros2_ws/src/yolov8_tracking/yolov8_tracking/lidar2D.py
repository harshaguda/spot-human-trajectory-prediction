## Transformations
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage, Image # Image is the message type sensor_msgs/msg/CompressedImage
from nav_msgs.msg import Odometry # nav_msgs/msg/Odometry
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

from scipy.spatial.transform import Rotation

import sensor_msgs_py.point_cloud2 as pc2

import matplotlib.pyplot as plt
IMG_H = 480
IMG_W = 640

class LidarCam():
    def __init__(self):
        # super().__init__('LidarCam')
        self.rslidar_to_body = np.array([
            [1.000,  0.000,  0.000,  -0.170],
            [0.000,  1.000,  0.000,  0.000],
            [0.000,  0.000,  1.000, 0.370],
            [0.000,  0.000,  0.000,  1.000]
        ])
        self.body_to_rs_lidar = np.linalg.inv(self.rslidar_to_body)

 

        self.back_fisheye_to_body = np.array([0.004, 0.273, -0.962, -0.419,
                                        1.000,  0.009,  0.007,  0.043,
            0.010, -0.962, -0.273,  0.013,
            0.000,  0.000,  0.000,  1.000
        ]).reshape((4, 4))

        self.left_fisheye_to_body = np.array([1.000 , 0.010  ,0.008 , 0.088,
                                                0.011 ,-0.293, -0.956 , 0.072,
                                                -0.007,  0.956 ,-0.293, -0.097,
                                                0.000,  0.000,  0.000 , 1.000]).reshape((4,4))
        
        self.right_fisheye_to_body = np.array([])

        self.body_to_back_fisheye = np.linalg.inv(self.back_fisheye_to_body)
        # print(body_to_back_fisheye, body_to_back_fisheye.shape)
        # body_to_back_fisheye = np.insert(body_to_back_fisheye,3,values=[0,0,0],axis=0)
        # self.body_to_back_fisheye = np.insert(body_to_back_fisheye,3,values=[0,0,0,1],axis=1)

        self.right_fisheye_to_body = np.array([1.000, -0.008,  0.001,  0.092,
                                                -0.003, -0.261 , 0.965, -0.059,
                                                -0.008, -0.965, -0.261, -0.099,
                                                0.000 , 0.000,  0.000,  1.000]).reshape((4,4))
        self.k_rfe = np.array([ 255.8085174560547, 0.0, 314.3297424316406, 0.0, 255.8085174560547, 240.5596466064453, 0.0, 0.0, 1.0]).reshape(3,3)

        # self.P_rfe = np.insert(self.k_rfe, 3, values=[-0.093, -0.111, 0.031], axis=1)
        self.P_rfe = np.insert(self.k_rfe, 3, values=[0, 0, 0], axis=1)




        self.camera_matrix = np.array([256.9691467285156, 0.0, 320.87396240234375, 0.0, 256.9691467285156, 240.7810516357422, 0.0, 0.0, 1.0]).reshape(3,3)
        self.P_rfe =  np.array([255.8085174560547,0.0,314.3297424316406,0.0,0.0,255.2574462890625,240.5596466064453,0.0,0.0,0.0,1.0,0.0]).reshape(3,4)
        back_tra = [0.07997186021560002, -0.00328245601495, 0.00046498809582900006]
        self.k_lfe = np.array([ 255.06076049804688, 0.0, 315.1103820800781, 0.0, 254.58456420898438, 238.33242797851562, 0.0, 0.0, 1.0]).reshape(3,3)
        # self.P_lfe = np.insert(self.k_lfe,3,values=[0.088,0.072, -0.097],axis=1)
        self.P_lfe = np.insert(self.k_lfe,3,values=[0,0, 0],axis=1)

        self.k_bfe = np.array([256.9691467285156, 0.0, 320.87396240234375, 0.0, 256.9691467285156, 240.7810516357422, 0.0, 0.0, 1.0]).reshape(3,3)

    #     x: 0.07494902694690002
    #   y: -0.00365773829895
    #   z: 0.0013797579456500003
        left_tra = [0, 0, 0]
        self.P_bfe = np.insert(self.camera_matrix ,3,values=back_tra,axis=1)
        self.IMG_W = 640
        self.IMG_H = 480
        print(self.P_bfe.shape)

        self.br = CvBridge()
        self.img = np.zeros((640,480))
        self.img_rfe = np.zeros((640,480))
        self.img_lfe = np.zeros((640,480))
        self.color = (255, 0, 0)

    def projection(self, img, points, rslidar, cam_transform, P):
        points = points @ rslidar.T
        #self.P_bfe.dot(self.body_to_back_fisheye.dot(self.points.T))
        # print(self.P_rfe.shape, self.left_fisheye_to_body.shape, self.)
        cam = P@cam_transform@points.T
        cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)
        # get u,v,z
        cam[:2] /= cam[2,:]
        
        # IMG_H,IMG_W = img.shape
        # restrict canvas in range
        # filter point out of canvas
        u,v,z = cam
        u_out = np.logical_or(u<0, u>IMG_W)
        v_out = np.logical_or(v<0, v>IMG_H)
        outlier = np.logical_or(u_out, v_out)
        cam = np.delete(cam,np.where(outlier),axis=1)

        lid2D = np.zeros((IMG_H, IMG_W), dtype=np.float64)
        for i, j, k in zip(u,v, z):
            if (i > IMG_W) or (j > IMG_H) or (i < 0) or (j < 0):
                continue

            lid2D[int(j)][int(i)] = k
            # print(k)
            cv2.circle(img, (int(i), int(j)), 1, self.color, -1)

        return cam, img, lid2D#np.repeat(lid2D,3).reshape(480,-1,3)
    def angle_point(self, x, K, IMG_W):
        foc_len = (K[0][0] +K[1][1])/2
        fov = 2*np.arctan(IMG_W/(2*K[0,0]))
        theta = (x- K[0][2])/foc_len
        # return theta
        return ((x - K[0,2])/IMG_W)*fov
    