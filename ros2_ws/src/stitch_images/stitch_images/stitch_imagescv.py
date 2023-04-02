# Basic ROS 2 program to stitch images from spot robot 

# Author:
# - Harsha Guda
# - 

import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage # Image is the message type sensor_msgs/msg/CompressedImage
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

from panoramaStitch import *

import numpy as np

class stitchImages(Node):
    def __init__(self):
        super().__init__('stitchImages')

        qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                depth=1
            )
        self.image_nodes = ['/spot/back_fisheye_image/compressed', '/spot/frontleft_fisheye_image/compressed', '/spot/frontright_fisheye_image/compressed',
                              '/spot/hand_color_image/compressed', '/spot/hand_image/compressed', '/spot/left_fisheye_image/compressed',
                              '/spot/right_fisheye_image/compressed']
        self.i = 0
        self.stitcher = cv2.Stitcher_create()
        # self.video = cv2.VideoWriter_fourcc()
        self.bfe = self.create_subscription(CompressedImage, self.image_nodes[0], self.listener_callback1, qos_profile=qos_profile)
        self.flfe = self.create_subscription(CompressedImage, self.image_nodes[1], self.listener_callback2, qos_profile=qos_profile)
        self.frfe = self.create_subscription(CompressedImage, self.image_nodes[2], self.listener_callback3, qos_profile=qos_profile)
        # self.hci = self.create_subscription(CompressedImage, self.image_nodes[3], self.listener_callback4, qos_profile=qos_profile)
        # self.hi = self.create_subscription(CompressedImage, self.image_nodes[4], self.listener_callback5, qos_profile=qos_profile)
        self.lfe = self.create_subscription(CompressedImage, self.image_nodes[5], self.listener_callback6, qos_profile=qos_profile)
        self.rfe = self.create_subscription(CompressedImage, self.image_nodes[6], self.listener_callback7, qos_profile=qos_profile)

        self.br = CvBridge()

    def listener_callback1(self, data):
        """
        Callback function.
        """
        # Display the message on the console
        # self.get_logger().info('Receiving video frame')
    
        # Convert ROS Image message to OpenCV image
        self.bfe_img = self.br.compressed_imgmsg_to_cv2(data)
        
        # current_frame = np.array(current_frame)

        # cv2.imshow('res', current_frame)
        # cv2.waitKey(1)
    def listener_callback2(self, data):
        """
        Callback function.
        """
        # Display the message on the console
        # self.get_logger().info('Receiving video frame')
    
        # Convert ROS Image message to OpenCV image
        self.flfe_img = self.br.compressed_imgmsg_to_cv2(data)
    
    def listener_callback3(self, data):
        """
        Callback function.
        """
        # Display the message on the console
        # self.get_logger().info('Receiving video frame')
    
        # Convert ROS Image message to OpenCV image
        self.frfe_img = self.br.compressed_imgmsg_to_cv2(data)

    def listener_callback6(self, data):
        """
        Callback function.
        """
        # Display the message on the console
        # self.get_logger().info('Receiving video frame')
    
        # Convert ROS Image message to OpenCV image
        self.lfe_img = self.br.compressed_imgmsg_to_cv2(data)
    
    def listener_callback7(self, data):
        """
        Callback function.
        """
        # Display the message on the console
        # self.get_logger().info('Receiving video frame')
    
        # Convert ROS Image message to OpenCV image
        self.rfe_img = self.br.compressed_imgmsg_to_cv2(data)
        self.show_images()
    
    def show_images(self):
        # concatenate image Horizontally
        # Hori1 = np.concatenate((self.bfe_img.reshape((640,480)), cv2.rotate(self.flfe_img, cv2.ROTATE_90_CLOCKWISE)), axis=1)
        
        # # concatenate image Vertically
        # Hori2 = np.concatenate((cv2.rotate(self.frfe_img, cv2.ROTATE_90_CLOCKWISE), self.lfe_img.reshape((640,480))), axis=1)
        image = cv2.rotate(self.rfe_img, cv2.ROTATE_180)
        images = []
        images.append(image)
        images.append(self.bfe_img)
        stitcher = cv2.Stitcher_create()
        # (status, stitched) = stitcher.stitch(images)
        # print(status)
        Hori1 = np.concatenate((image, self.bfe_img, self.lfe_img), axis=1)
        vid_path = 'video/' + str(self.i) + '.jpg'
        # print(vid_path)
        #cv2.imwrite(vid_path, Hori1)
        #self.i += 1
        # concatenate image Vertically
        Hori2 = np.concatenate((self.frfe_img, self.flfe_img), axis=1)

        im2 = cv2.rotate(self.frfe_img, cv2.ROTATE_90_CLOCKWISE)
        im3 = cv2.rotate(self.flfe_img, cv2.ROTATE_90_CLOCKWISE)
        ccn = np.concatenate((im2, im3), axis=1)
        # cv2.imwrite('3.jpg', im2)
        # cv2.imwrite('4.jpg', im3)

        # exit()
        # Reading the 2 images.
        # Image1 = cv2.imread("InputImages/Sun/1.jpg")
        # Image2 = cv2.imread("InputImages/Sun/2.jpg")

        # Calling function for stitching images.
        # StitchedImage = StitchImages(self.bfe_img, self.lfe_img)

        # Displaying the stitched images.
        # if not status:
        #     cv2.imshow('img',stitched)
        # plt.show()
        # total = np.concatenate((Hori1, Hori2), axis=0)
        Hori2 = np.concatenate((im2, im3), axis=1)

        cv2.imshow('im1', Hori1)
        cv2.imshow('im2', Hori2)
        # cv2.imshow('res', ccn)

        # image = cv2.rotate(self.rfe_img, cv2.ROTATE_180)
        # cv2.imshow('1', Hori1)
        cv2.waitKey(1)


def main(args=None):

    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    stichimgs = stitchImages()

    # Spin the node so the callback function is called.
    rclpy.spin(stichimgs)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    stichimgs.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
