import argparse
import cv2
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

from sensor_msgs.msg import CompressedImage, Image # Image is the message type sensor_msgs/msg/CompressedImage

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray


import sensor_msgs_py.point_cloud2 as pc2

from lidar2D import LidarCam
from rosbags.highlevel import AnyReader
from helper_pointcloud import *
from std_msgs.msg import Header

# from rospy_message_converter import message_converter
import json
from std_msgs.msg import String
# dictionary = { 'data': 'Howdy' }
# message = message_converter.convert_dictionary_to_ros_message('std_msgs/String', dictionary)
# https://github.com/DFKI-NI/rospy_message_converter/tree/humble
# from rclpy_message_converter import json_message_converter, message_converter
# from std_msgs.msg import String

from geometry_msgs.msg import TransformStamped

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

class ROSTracker(Node):
    

    @torch.no_grad()
    def __init__(
            self,
            source='0',
            yolo_weights=WEIGHTS / 'best.pt',  # model.pt path(s),
            reid_weights=WEIGHTS / 'lmbn_n_cuhk03_d.pt',  # model.pt path,
            tracking_method='strongsort',
            tracking_config=None,
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            show_vid=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            save_trajectories=False,  # save trajectories for each track
            save_vid=False,  # save confidences in --save-txt labels
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=True,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs' / 'track',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            hide_class=False,  # hide IDs
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
            retina_masks=False,
    ):
        super().__init__('ROSTracker')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=1
        )
        
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        self.pub = self.create_publisher(String, '/trajectories', 10)
        self.lidar_pub = self.create_publisher(PointCloud2, '/rslidar_points', 100)

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.save_trajectories = save_trajectories
        self.save_vid = save_vid
        self.save_crop = save_crop
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.hide_class = hide_class
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.line_thickness = line_thickness
        self.classes = classes
        self.half = half
        self.br = CvBridge()
        self.tracking_method = tracking_method
        self.tracking_config = ROOT / 'trackers' / self.tracking_method / 'configs' / (self.tracking_method + '.yaml')
        self.reid_weights = reid_weights
        self.source = str(source)
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.save_img = not nosave and not source.endswith('.txt')  # save inference images
        self.is_file = Path(source).suffix[1:] in (VID_FORMATS)
        self.is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if self.is_url and self.is_file:
            source = check_file(source)  # download
        self.show_vid = show_vid
        self.retina_masks = retina_masks
        self.rfe_img = np.zeros((480,640))
        self.dangles = []
        # self.ld_img = np.zeros((480,640))
        self.ld2D = LidarCam()
        self.frame_id_rfe = 0
        self.frame_id_bfe = 0
        self.frame_id_lfe = 0
        self.k_rfe = np.array([ 255.8085174560547, 0.0, 314.3297424316406, 0.0, 255.8085174560547, 240.5596466064453, 0.0, 0.0, 1.0]).reshape(3,3)
        self.k_bfe = np.array([256.9691467285156, 0.0, 320.87396240234375, 0.0, 256.9691467285156, 240.7810516357422, 0.0, 0.0, 1.0]).reshape(3,3)
        self.k_lfe = np.array([ 255.06076049804688, 0.0, 315.1103820800781, 0.0, 254.58456420898438, 238.33242797851562, 0.0, 0.0, 1.0]).reshape(3,3)
        self.right_cam_angle = -1.834
        self.left_cam_angle = 1.869
        self.back_cam_angle = 3.117
        # https://towardsdatascience.com/understanding-homography-a-k-a-perspective-transformation-cacaed5ca17
        # Directories
        if not isinstance(yolo_weights, list):  # single yolo model
            exp_name = yolo_weights.stem
        elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
            exp_name = Path(yolo_weights[0]).stem
        else:  # multiple models after --yolo_weights
            exp_name = 'ensemble'
        exp_name = name if name else exp_name + "_" + reid_weights.stem
        save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
        (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.device = select_device(device)
        self.is_seg = '-seg' in str(yolo_weights)
        self.model = AutoBackend(yolo_weights, device=self.device, dnn=dnn, fp16=half)
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_imgsz(imgsz, stride=stride)  # check image size

        # Dataloader
        self.bs = 1
        
        vid_path, vid_writer, txt_path = [None] * self.bs, [None] * self.bs, [None] * self.bs
        self.model.warmup(imgsz=(1 if pt or self.model.triton else self.bs, 3, *imgsz))  # warmup

        # Create as many strong sort instances as there are video sources
        self.tracker_list = []
        for i in range(self.bs):
            self.tracker = create_tracker(self.tracking_method, self.tracking_config, self.reid_weights, self.device, self.half)
            self.tracker_list.append(self.tracker, )
            if hasattr(self.tracker_list[i], 'model'):
                if hasattr(self.tracker_list[i].model, 'warmup'):
                    self.tracker_list[i].model.warmup()
        self.outputs = [None] * self.bs

        # Run tracking
        #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile(), Profile())
        self.curr_frames, self.prev_frames = [None] * self.bs, [None] * self.bs
        
        
    def timer_callback(self):
        self.rb()
    
    def cylindrical_warp(self, img,K):
        """
        cylindrical_warp function is used to convert the fisheye image captured by spot 
        robot to convert into a cylindrical image to remove the distortions created by
        fisheye view.

        Inputs:
        img: cv2 image.
        K: camera intrinsic property, can be retrieved from the /spot/<camera>_camera_info topic
        """
        foc_len = (K[0][0] +K[1][1])/2
        cylinder = np.zeros_like(img)
        temp = np.mgrid[0:img.shape[1],0:img.shape[0]]
        x,y = temp[0],temp[1]
        color = img[y,x]
        theta= (x- K[0][2])/foc_len # angle theta
        h = (y-K[1][2])/foc_len # height
        p = np.array([np.sin(theta),h,np.cos(theta)])
        p = p.T
        p = p.reshape(-1,3)
        image_points = K.dot(p.T).T
        points = image_points[:,:-1]/image_points[:,[-1]]
        points = points.reshape(img.shape[0],img.shape[1],-1)
        cylinder = cv2.remap(img, (points[:, :, 0]).astype(np.float32), (points[:, :, 1]).astype(np.float32), cv2.INTER_LINEAR)
        return cylinder

    def planar(self, img, K):
        foc_len = (K[0][0] +K[1][1])/2
        cylinder = np.zeros_like(img)
        temp = np.mgrid[0:img.shape[1],0:img.shape[0]]
        x,y = temp[0],temp[1]
    
    def stitchimages(self, lfe, bfe, rfe, img=False):
        """
        stitchimages functions takes in three images, performs cylindrical warp and concateantes
        them together. It works with both images and Lidar2D.
        Inputs:
        lfe: left fisheye image
        bfe: back fisheye image
        rfe: right fisheye image
        img: Boolean it is used to specify if the given inputs are whether cv2 images or the 
        Lidar2D
        """
        img_cyl_lfe = self.cylindrical_warp(lfe, self.k_lfe)
        img_cyl_bfe = self.cylindrical_warp(bfe, self.k_bfe)
        img_cyl_rfe = self.cylindrical_warp(rfe, self.k_rfe)
        i1 = img_cyl_lfe[:,89:546]
        i2 = img_cyl_bfe[:,95:529]
        img_cyl_rfe = cv2.rotate(img_cyl_rfe, cv2.ROTATE_180)
        i3 = img_cyl_rfe[:,95:546]
        # print(img, i3.dtype, i2.dtype)
      
        if img:
            self.cn = np.concatenate((i3, i2 ,i1), axis=1, dtype=np.uint8) 

        else:
            
            self.cn = np.concatenate((i3, i2 ,i1), axis=1) 
            # print(self.cn.dtype)
            crp_nz = self.cn[np.nonzero(self.cn)]
            # print('crp: ', crp_nz.flatten())
        return self.cn
    

    def make_transforms(self, frame_id, child_id, translation, quat):
        """
        This function is used to publish the tf frame, which is required to visualise the 
        Lidar pointclouds.
        Taken from: https://docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Static-Broadcaster-Py.html
        """
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_id

        t.transform.translation.x = float(translation.x)
        t.transform.translation.y = float(translation.y)
        t.transform.translation.z = float(translation.z)
        
        t.transform.rotation.x = quat.x
        t.transform.rotation.y = quat.y
        t.transform.rotation.z = quat.z
        t.transform.rotation.w = quat.w

        self.tf_static_broadcaster.sendTransform(t)
    
    def rb(self):
        # with AnyReader([Path('/home/spot/Downloads/rosbag2_2023_02_14-17_16_36')]) as reader:
        with AnyReader([Path('/home/spot/Downloads/rosbag2_2023_05_23-17_24_15')]) as reader:
        # with AnyReader([Path('/home/spot/rosbags/rosbag2_2023_05_23-14_02_50')]) as reader:
            connections = [x for x in reader.connections if x.topic in ['/tf_static', '/rslidar_points', '/spot/left_fisheye_image/compressed', '/spot/right_fisheye_image/compressed', '/spot/back_fisheye_image/compressed']] #,in [ '/rslidar_points']
            
            read_lidar, read_img, lfe_f, rfe_f, bfe_f = False, False, False, False, False
            ld2cam = LidarCam()
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                # lidar = reader.deserialize(rawdata, connection.msgtype)
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                if connection.topic == '/spot/left_fisheye_image/compressed':
                    self.lfe_img = cv2.imdecode( msg.data, cv2.IMREAD_ANYCOLOR )
                    
                    self.frame_id_lfe += 1
                    lfe_f = True
                elif connection.topic == '/spot/right_fisheye_image/compressed':
                    self.rfe_img = cv2.imdecode( msg.data, cv2.IMREAD_ANYCOLOR )
                    self.frame_id_rfe += 1
                    rfe_f = True

                elif connection.topic == '/spot/back_fisheye_image/compressed':
                    self.bfe_img = cv2.imdecode( msg.data, cv2.IMREAD_ANYCOLOR )
                    self.frame_id_bfe += 1
                    bfe_f = True
                    
                    read_img=True
                elif connection.id == 48:
                    # print(msg)
                    for msg_transform in msg.transforms:
                        self.make_transforms(msg_transform.header.frame_id,msg_transform.child_frame_id, msg_transform.transform.translation, msg_transform.transform.rotation)
                       
                
                elif connection.topic == '/rslidar_points':

                    dtype = np.float32
                    itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.
                    
                    kwargs1 = {'name': 'x', 'offset': 0, 'datatype':PointField.FLOAT32, 'count':1}
                    kwargs2 = {'name': 'y', 'offset': 4, 'datatype':PointField.FLOAT32, 'count':1}
                    kwargs3 = {'name': 'z', 'offset': 8, 'datatype':PointField.FLOAT32, 'count':1}
                    kwargs4 = {'name': 'intensity', 'offset': 16, 'datatype':PointField.FLOAT32, 'count':1}

                    fields = [PointField(**kwargs1),
                                PointField(**kwargs2),
                                PointField(**kwargs3),
                                PointField(**kwargs4),
                                ]
                    header = Header()
                    header.stamp = self.get_clock().now().to_msg()
                    header.frame_id = 'rslidar'
                
                    pc = read_points(msg, field_names = ['x','y','z','intensity'], skip_nans=True)
                    cloud_points = []
                    for p in pc:
                        cloud_points.append(list(p))
                    
                    points = np.array(cloud_points)

                    pc_pub = pc2.create_cloud(header, fields, points)
                    self.lidar_pub.publish(pc_pub)

                    points = points[:, :3]
                # Add a column of ones
                    points = np.hstack([points, np.ones((points.shape[0], 1))])
                    read_lidar = True
                # if ((self.frame_id_lfe + self.frame_id_bfe + self.frame_id_rfe) % 2 == 0 )& read_lidar & (self.frame_id_lfe != 0):
                if lfe_f & bfe_f &  rfe_f & read_lidar:
                    read_lidar = False
                    lfe_f = False
                    bfe_f = False
                    rfe_f = False
                    cn_img = self.stitchimages(self.lfe_img, self.bfe_img, self.rfe_img, img=True)
                    self.callback(cn_img, points)
                    # self.cam, _ = self.ld2D.projection(self.lfe_img, points, self.ld2D.rslidar_to_body, self.ld2D.left_fisheye_to_body, self.ld2D.P_lfe)
                    self.cam, _, self.ld_lfe = ld2cam.projection(self.lfe_img, points, ld2cam.rslidar_to_body, ld2cam.left_fisheye_to_body, ld2cam.P_lfe )
                    self.cam, _, self.ld_bfe  = ld2cam.projection(self.bfe_img, points, ld2cam.rslidar_to_body, ld2cam.body_to_back_fisheye, ld2cam.P_bfe )
                    self.cam, _, self.ld_rfe = ld2cam.projection(self.rfe_img, points, ld2cam.rslidar_to_body, ld2cam.right_fisheye_to_body, ld2cam.P_rfe)
                    # crp_nz = self.ld_bfe[np.nonzero(self.ld_bfe)]
                    # print(crp_nz.flatten())
                    self.ld_img = self.stitchimages(self.ld_lfe, self.ld_bfe, self.ld_rfe)
                    
                    # cv2.imshow('img', self.ld_img)
                    # cv2.waitKey(1)
    @torch.no_grad()
    def callback(self, rfe_img, points):
        self.rfe_img = rfe_img
        # self.rfe_img = cv2.resize(self.rfe_img, (640,640), interpolation=cv2.INTER_AREA)
        im = np.stack((self.rfe_img, self.rfe_img, self.rfe_img), axis=0)
        
                
        im0s = np.repeat(self.rfe_img,3).reshape(480,-1,3)
        visualize = False
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            if len(im.shape) == 3:
                im = im[None] 
            preds = self.model(im, augment=self.augment, visualize=visualize)

        # Apply NMS
        with self.dt[2]:
            if self.is_seg:
                masks = []
                p = non_max_suppression(preds[0], self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, max_nms=32)
                proto = preds[1][-1]
            else:
                p = non_max_suppression(preds, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            
        # Process detections
        for i, det in enumerate(p):  # detections per image
            self.seen += 1
            im0 = im0s.copy()
            self.curr_frames[i] = im0
            # video file
            
            # s += '%gx%g ' % im.shape[2:]  # print string
           
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            # ld_an = Annotator(self.ld_img, line_width=self.line_thickness, example=str(self.names))
            if hasattr(self.tracker_list[i], 'tracker') and hasattr(self.tracker_list[i].tracker, 'camera_update'):
                if self.prev_frames[i] is not None and self.curr_frames[i] is not None:  # camera motion compensation
                    self.tracker_list[i].tracker.camera_update(self.prev_frames[i], self.curr_frames[i])

            if det is not None and len(det):
                if self.is_seg:
                    shape = im0.shape
                    # scale bbox first the crop masks
                    if self.retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                        masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                    else:
                        masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                
               

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with self.dt[3]:
                    self.outputs[i] = self.tracker_list[i].update(det.cpu(), im0)
                
                # draw boxes for visualization
                if len(self.outputs[i]) > 0:
                    
                    if self.is_seg:
                        # Mask plotting
                        annotator.masks(
                            masks[i],
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if self.retina_masks else im[i]
                        )
                        
                    traj_dict = dict()
                    for j, (output) in enumerate(self.outputs[i]):
                        
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]

                        

                        if self.save_vid or self.save_crop or self.show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if self.hide_labels else (f'{id} {self.names[c]}' if self.hide_conf else \
                                (f'{id} {conf:.2f}' if self.hide_class else f'{id} {self.names[c]} {conf:.2f}'))
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)
                           
                            p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                            x1, y1 = p1
                            x2, y2 = p2
                            
                            crp = self.ld_img[y1:y2, x1:x2]
                            crp_nz = crp[np.nonzero(crp)]
                            m = crp[np.where((crp < 4))].max()
                            
                            xm, ym = int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)
                            
                            if xm <= 451:
                                xm = (xm/451)*640
                                # print('1 ', xm)
                                distance = crp[np.where((crp > 1) & (crp < 4))].mean() -0.1
                                angle_o = self.ld2D.angle_point( xm, self.ld2D.k_rfe, self.ld2D.IMG_W) 
                                
                                angle = -angle_o + self.right_cam_angle + 0.3 #+ np.pi/2
                            if (xm >451) & (xm <= 885):
                                
                                xm = ((xm - 451)/434)*640
                                # print('2 ', xm)
                                distance = crp[np.where((crp > 1) & (crp < 4))].mean() - 0.2#+ 0.54
                                angle_o = self.ld2D.angle_point( xm, self.ld2D.k_bfe, self.ld2D.IMG_W)
                                angle = -angle_o + self.back_cam_angle
                            if xm > 885:
                                xm = ((xm - 885)/457)*640
                                distance = crp[np.where((crp > 1) & (crp < 4))].mean() -0.3
                                # print('3 ', xm)
                                angle_o = self.ld2D.angle_point( xm, self.ld2D.k_lfe, self.ld2D.IMG_W)
                                angle = -angle_o + self.left_cam_angle #- np.pi/2


                            # _, _, d=np.take(self.cam,np.where(outlier),axis=1).reshape(3, -1)
                            traj_dict[id] = {'distance':distance, 'angle':angle}
                            self.dangles.append({'distance':distance, 'angle':angle})
                            np.save('dangles.npy', np.array(self.dangles))
                            print(traj_dict, 'angle o ', angle_o)
                            # traj_dict[id] = {'distance':2.964, 'angle':3.0469}

                            
                            if self.save_trajectories and self.tracking_method == 'strongsort':
                                q = output[7]
                                self.tracker_list[i].trajectory(im0, q, color=color)
                        # data = message_converter.convert_dictionary_to_ros_message('std_msgs/msg/String', traj_dict)
                        msg = String()
                        # print(traj_dict, 'Angle o ', angle_o)
                        msg.data = json.dumps(traj_dict)
                        self.pub.publish(msg)
            else:
                pass
                #tracker_list[i].tracker.pred_n_update_all_tracks()
                
            # Stream results
            im0 = annotator.result()
            # ld2d = ld_an.result()
            if self.show_vid:
               
                cv2.namedWindow(str('sth'), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str('sth'), im0.shape[1], im0.shape[0])
               
                cv2.imshow(str('sth'), im0)
                # cv2.imshow('LiDar', ld2d)
               

                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

        

            self.prev_frames[i] = self.curr_frames[i]







def main(args=None):

    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    stichimgs = ROSTracker()

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