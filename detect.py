# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

from PIL.Image import Image
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox
from src.server import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

class Detect:
    def __init__(self, **kwargs):
        # self.img = img
        self.model=kwargs.get('weights', 'yolov5x.pt')
        self.device=kwargs.get('device', None)
        self.imgsz=kwargs.get('imgsz', 640)
        self.dnn=kwargs.get('dnn', False)
        self.half=kwargs.get('half', False)
        self.augment=kwargs.get('augment', False)
        self.visualize=kwargs.get('visualize', False)

                # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.model, device=self.device, dnn=self.dnn)

        self.model.eval()

    def convertImage(self, image, stride=32, auto=True):
        # img0 = cv2.imread('src/in3.jpeg') 
        # image = img0

        assert image is not None, f'Image Not Found'
        stride = 640
        # auto = False
        img = letterbox(image, stride, auto)[0]
        
        img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img

    @torch.no_grad()
    def run(self, image):

        stride, pt, jit, onnx, engine = self.model.stride, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Half
        half = self.half & (pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if half else self.model.model.float()

        img = self.convertImage(image, stride=stride, auto=pt)

        # Run inference
        dt = []
        # self.model.warmup(imgsz=(1, 3, *self.imgsz), half=half)  # warmup
        t1 = time_sync()
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        batch_image = 50
        im = im.repeat(batch_image, 1, 1, 1)
        print('data shape is ',im.shape)
        
        t2 = time_sync()
        dt.append(t2 - t1)

        # Inference
        self.model(im, augment=self.augment, visualize=self.visualize)

        t3 = time_sync()
        dt.append(t3 - t2)

        # Print results
        t = tuple(x * 1E3 for x in dt)  
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference' % t)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

######################## server part start ########################

from multiprocessing import Process,Queue
import time,random,os
def consumer(q, detect, client_num):
    t1 = time_sync()
    for i in range(20):
        frame=q.get() 

        if frame is None:
            break

        detect.run(frame)

    t2 = time_sync()
    print('server端总时长{:.3f}s'.format(t2-t1))

# 进程最小函数
def producer(name,q):
    发送图片间隔 = 1.0/3.0
    t = time.time()
    for i in range(5):
        time.sleep(发送图片间隔)

        image = cv2.imread('src/in3.jpeg')
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)

        q.put(image)
    
    print('单个client发送图片 time usage {:.3f}s'.format(time.time() - t))

######################## server part end ########################

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    
    # 创建推理模型
    detect = Detect(**vars(opt))

    # 推理图片队列
    image_queue = Queue()
    
    clients = []
    client_num = 4
    # 创建client进程
    for i in range(client_num):
        clients.append(
            Process(target=producer,args=('client'+str(i), image_queue))
        )

    # 启动client进程
    for client in clients:
        client.start()

    # 启动server（本进程即为server进程）
    consumer(image_queue, detect, client_num)

    # join client进程
    for client in clients:
        client.join()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)