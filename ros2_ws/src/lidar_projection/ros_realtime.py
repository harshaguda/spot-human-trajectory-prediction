import rclpy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2

from show import *
from rclpy.node import Node
import cv2

class Lidar2D(Node):
	def __init__(self):
		super().__init__('lidar2d')
		self.subscription = self.create_subscription(
			PointCloud2,
			'/rslidar_points',
			self.on_new_point_cloud,
			10)
		self.subscription  # prevent unused variable warning
		self.HRES = 0.35         # horizontal resolution (assuming 20Hz setting)
		self.VRES = 1.33          # vertical res
		self.VFOV = (-30.67, 10.67) # Field of view (-ve, +ve) along vertical axis
		self.Y_FUDGE = 5         # y fudge factor for velodyne HDL 64E
		self.im=0
	def on_new_point_cloud(self, data):
		global im 
		
		pc = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z","intensity"))
		npc = np.zeros((6746,4))
		
		cloud_points = []
		for p in pc:
			# print(p)
			cloud_points.append(list(p))
		# for p in range(4):
		# 	npc[:,p] = pc[p:6746*(p+1)]
		npc = np.array(cloud_points)
		#lidar_to_2d_front_view(npc, v_res=VRES, h_res=HRES, v_fov=VFOV, val="depth", y_fudge=Y_FUDGE)
		#lidar_to_2d_front_view(npc, v_res=VRES, h_res=HRES, v_fov=VFOV, val="height", y_fudge=Y_FUDGE)
		#lidar_to_2d_front_view(npc, v_res=VRES, h_res=HRES, v_fov=VFOV, val="reflectance", y_fudge=Y_FUDGE)
		#im = birds_eye_point_cloud(npc, side_range=(-10, 10), fwd_range=(-10, 10), res=0.1)
		# print(npc.shape)
		self.im = point_cloud_to_panorama(npc,
								v_res=self.VRES,
								h_res=self.HRES,
								v_fov=self.VFOV,
								y_fudge=5,
								d_range=(0,100))
		im = self.im
		print(im.shape)
		cv2.imshow('image',im)
		cv2.waitKey(1)
		

def main(args=None):
	rclpy.init(args=args)
	lidar2d = Lidar2D()
	print('Here')
	rclpy.spin(lidar2d)
	lidar2d.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()



