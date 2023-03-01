# Basic ROS 2 program to subscribe to real-time streaming 
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
  
# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage # Image is the message type sensor_msgs/msg/CompressedImage
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
from ultralytics import YOLO

import numpy as np

class ImageSubscriber(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_subscriber')
    qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

    # Create the subscriber. This subscriber will receive an Image
    # from the spot/back_fisheye_image/compressed topic. The queue size is 1 messages.
    self.subscription = self.create_subscription(
      CompressedImage, 
      '/spot/back_fisheye_image/compressed',  # Topic from which we need images.
      self.listener_callback, 
      qos_profile=qos_profile)
    self.subscription # prevent unused variable warning
      
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()
    self.model = YOLO("yolov8n.pt") 


   
  def listener_callback(self, data):
    """
    Callback function.
    """
    # Display the message on the console
    self.get_logger().info('Receiving video frame')
 
    # Convert ROS Image message to OpenCV image
    current_frame = self.br.compressed_imgmsg_to_cv2(data)
    # frame = cv2.imread('bus.jpg')
    # current_frame = current_frame[:,:,::-1]
    current_frame = np.array(current_frame)
    frame = np.stack((current_frame, current_frame, current_frame), axis=-1).reshape((current_frame.shape[0], current_frame.shape[1], 3))
    
    # Predict with the model
    results = self.model.track(frame, show=True)  # predict on an image 

    # results = self.model.track(source=current_frame, show=True)
    # Display image
    # cv2.imshow("camera", frame)
    
    # cv2.waitKey(1)
  
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  image_subscriber = ImageSubscriber()
  
  # Spin the node so the callback function is called.
  rclpy.spin(image_subscriber)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_subscriber.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()