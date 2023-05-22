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
        super().__init__('stitchImages')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=1
        )
        self.lidar = self.create_subscription(PointCloud2,'/rslidar_points', self.get_lidar,10)
        # self.subscription = self.create_subscription(
        # CompressedImage,
        # '/spot/left_fisheye_image/compressed',
        # self.callback,
        # qos_profile = qos_profile)

        self.subscription = self.create_subscription(
        Image,
        '/spot/stitchedImages',
        self.callback,
        qos_profile = qos_profile)
        self.pub = self.create_publisher(Float32MultiArray, '/trajectories', 10)
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
        self.ld_img = np.zeros((480,640))
        self.ld2D = LidarCam()

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
        print('device ',self.device)
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
        
        # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
        # if save_txt or save_vid:
        #     s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # if update:
        #     strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)

    # for frame_idx, batch in enumerate(dataset):
    def get_lidar(self, data):
        points = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z","intensity"))
        print(points.shape)
        cloud_points = []
        for p in points:
            # print(p)
            cloud_points.append(list(p))
        
        points = np.array(cloud_points)
        # print(points.shape)
        # print(points.shape)
        points = points[:, :3]
        # Add a column of ones
        points = np.hstack([points, np.ones((points.shape[0], 1))])
        # Transform points
        # self.projection(self.img, points, self.rslidar_to_body, self.body_to_back_fisheye, self.P_bfe)
        self.cam, _ = self.ld2D.projection(self.rfe_img, points, self.ld2D.rslidar_to_body, self.ld2D.left_fisheye_to_body, self.ld2D.P_lfe)
        # self.cam, _, self.ld_img = self.ld2D.projection(self.rfe_img, points, self.ld2D.rslidar_to_body, self.ld2D.body_to_back_fisheye, self.ld2D.P_bfe)

        # self.ld2d = np.zeros((480,640,3))
        # for i in range(3):
        #     self.ld2d[:,:,i] = ld_img
    def cylindrical_warp(self, img,K):
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
    
    @torch.no_grad()
    def callback(self, data):
        # print('callback')
        # self.rfe_img = self.br.compressed_imgmsg_to_cv2(data)
        self.rfe_img = self.br.imgmsg_to_cv2(data)

        # self.rfe_img = cv2.resize(self.rfe_img, (640,640), interpolation=cv2.INTER_AREA)
        frame = np.stack((self.rfe_img, self.rfe_img, self.rfe_img), axis=0)
        # print(frame.shape)
        # frame = cv2.resize(frame, (640,640), interpolation = cv2.INTER_AREA)
        # print(frame.shape)
        # print(frame.shape)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)
        # print(frame.shape)
        im = frame
        # im0s = np.zeros((480,640,3), dtype=np.int8)
        im0s = np.zeros((480,1342,3), dtype=np.int8)

        # print(im0s.shape)
        for i in range(3):
            im0s[:,:,i] = frame[i]
        im0s = np.array(im0s, dtype=np.uint8)
        # cv2.imshow('sss', im0s)
        # cv2.waitKey(0)
        # visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
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
            # print('shape',im.shape)
            preds = self.model(im, augment=self.augment, visualize=visualize)

        # Apply NMS
        with self.dt[2]:
            if self.is_seg:
                masks = []
                p = non_max_suppression(preds[0], self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)
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
            ld_an = Annotator(self.ld_img, line_width=self.line_thickness, example=str(self.names))
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
                
                # print(det)

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
                            print(label, id)
                            ld_an.box_label(bbox, label, color=color)
                            p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                            xm, ym = int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)
                            angle = self.ld2D.angle_point( xm, self.ld2D.k_lfe, self.ld2D.IMG_W) -1.869#-np.pi# 3.117
                            cv2.imshow('crp', self.ld_img[ym-15:ym+15, xm-15:xm+15])
                            u, v, z = self.cam
                            u_out = np.logical_and(u<xm+10, u>xm-10)
                            v_out = np.logical_and(v<ym+10, v>ym-10)
                            outlier = np.logical_and(u_out, v_out)

                            dist = self.ld_img[ym-15:ym+15, xm-15:xm+15]
                            nnz = dist[np.nonzero(dist)]
                            _, _, d=np.take(self.cam,np.where(outlier),axis=1).reshape(3, -1)
                            print('Distance ',nnz.mean(), np.mean(d) )

                            msg = Float32MultiArray()
                            print([id, nnz.mean(), angle])
                            msg.data = [float(id), np.mean(d), angle]
                            self.pub.publish(msg)
                            # cv2.imshow('crp', self.ld_img[p1[1]:p2[1], p1[0]:p2[0]])
                            if self.save_trajectories and self.tracking_method == 'strongsort':
                                q = output[7]
                                self.tracker_list[i].trajectory(im0, q, color=color)
                            

            else:
                pass
                #tracker_list[i].tracker.pred_n_update_all_tracks()
                
            # Stream results
            im0 = annotator.result()
            ld2d = ld_an.result()
            if self.show_vid:
                # print(show_vid)
                # print(p)
                # if platform.system() == 'Linux' and p not in self.windows:
                #     self.windows.append(p)
                cv2.namedWindow(str('sth'), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str('sth'), im0.shape[1], im0.shape[0])
                # print(p, im0)
                cv2.imshow(str('sth'), im0)
                cv2.namedWindow(str('ld'), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str('ld'), im0.shape[1], im0.shape[0])
                cv2.imshow('ld', ld2d)
                print('ld', self.ld_img.shape)

                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

        

            self.prev_frames[i] = self.curr_frames[i]
            
        # Print total time (preprocessing + inference + NMS + tracking)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in self.dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

        # Print results


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov8s-seg.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'lmbn_n_cuhk03_d.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt





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