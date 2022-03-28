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
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import multiprocessing
from multiprocessing import Process
import os
import sys
from pathlib import Path
from unittest import defaultTestLoader
from PIL.Image import Image
import numpy as np
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox, batch_letterbox
from encode_decode import encode_dict, decode_dict
import socket
import socketserver
from controller_test import Controller

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from multiprocessing import Queue
from multiprocessing import Manager

class Detect:
    def __init__(self, **kwargs):
        self.model = kwargs.get('weights', 'yolov5x.pt')
        self.device = kwargs.get('device', 'cpu')
        self.imgsz = kwargs.get('imgsz', 640)
        self.dnn = kwargs.get('dnn', False)
        self.half = kwargs.get('half', False)
        self.augment = kwargs.get('augment', False)
        self.visualize = kwargs.get('visualize', False)
        self.sequence = kwargs.get('sequence', False)
        
        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.model, device=self.device, dnn=self.dnn)
        self.model.eval()
    
    def convertImage(self, image, stride=32, auto=True):
        
        assert image is not None, f'Image Not Found'
        stride = 640
        # auto = False
        # img = batch_letterbox(image, stride, auto)[0]
        # img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
        img = batch_letterbox(image, stride, auto)
        img = img.transpose((0, 3, 1, 2))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        return img
    
    @torch.no_grad()
    def run(self, image):
        self.model.to('cuda')
        stride, pt, jit, onnx, engine = self.model.stride, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        # Half
        half = self.half & (
                    pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if half else self.model.model.float()
        # crop img to specific size
        img = self.convertImage(image, stride=stride, auto=pt)
        
        # Run inference
        dt = []
        t1 = time_sync()
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        print('data shape is ', im.shape)
        
        t2 = time_sync()
        dt.append(t2 - t1)
        
        # Inference
        self.model(im, augment=self.augment, visualize=self.visualize)
        t3 = time_sync()
        dt.append(t3 - t2)
        
        # Print results
        t = tuple(x * 1E3 for x in dt)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference' % t)
        self.model.to('cpu')


class Controller:
    """
    作为Controller, 监视所有的detector
    """
    
    def __init__(self, detector_num=1, img_num=100, cmdQueues=None, imgQueues=None):
        self.act_id = 0  # active task id
        self.detector_num = detector_num
        self.img_num = img_num
        self.cmdQueues = cmdQueues
        self.imgQueues = imgQueues
        self.exitSignal = 0  # when all workers finished job, exitSignal == detector_num
    
    def initQueues(self):
        for i in range(self.img_num):
            for q in self.imgQueues:
                q.put(np.zeros(shape=(1920, 1080, 3)))
                
        self.cmdQueues[self.act_id].put('infer')
        return
    
    def update_cmd_queue(self):
        """
        根据当前的self.controller_state, 对所有的detector进行控制
        """
        # for i in range(detector_num):
        print('***  check ctrl cmd queue size : {}:{}  ***'.format(self.act_id, self.cmdQueues[self.act_id].qsize()))
        # time.sleep(10)
        if self.cmdQueues[self.act_id].empty() is False:
            cmd = self.cmdQueues[self.act_id].get()
            print('worker {} current state : {}'.format(self.act_id, cmd))
            if cmd == 'done':
                if self.act_id + 1 == self.detector_num:
                    self.act_id = 0
                else:
                    self.act_id += 1
                
                self.cmdQueues[self.act_id].put('infer')
            if cmd == 'exit':
                self.exitSignal += 1       
        return

    def run(self):
        self.initQueues()
        while True:
            self.update_cmd_queue()
            if self.exitSignal == self.detector_num:
                print("all workers has finished jobs !")
                break
        return
            

class Worker:

    def __init__(self, id, img_num, batchsize, engine: Detect, cmdQueue, imgQueue, time_stamp):
        self.id = id
        self.img_num = img_num
        self.batchsize = batchsize
        self.engine = engine
        self.cmdQueue = cmdQueue
        self.imgQueue = imgQueue
        self.time_stamp = time_stamp
    
    def run(self):
        self.time_stamp.append(time.time())
        while True:
            if self.imgQueue.empty() is True:
                print('all the images have been processed by worker {}'.format(self.id))
                self.cmdQueue.put('exit')
                break
            
            if self.cmdQueue.empty() is False and self.cmdQueue.get() == 'infer':
                frames = []
                for _ in range(self.batchsize):
                    if self.imgQueue.qsize() > 0:
                        frames.append(self.imgQueue.get())    
                pred = self.engine.run(np.stack(frames))
                
                # send signal to controller
                self.cmdQueue.put('done')
                print("worker {} send inferring finished signal, queue size: {} ".format(self.id, self.cmdQueue.qsize()))
                
        self.time_stamp.append(time.time())
        return              

def runWorker(detector):
    detector.run()

def runController(controller):
    controller.run()

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
    parser.add_argument('--bs', type=int, default=1, help='batch size of img')
    parser.add_argument('--img_num', type=int, default=50, help='the number of img sent by each client')
    parser.add_argument('--videos', type=int, default=8, help='the number of video stream')
    parser.add_argument('--workers', type=int, default=1, help='the number of detector')
    # parser.add_argument('--sequence', action='store_true', help='whether run 5s followed by 5x')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    multiprocessing.set_start_method('spawn')
    
    # initialize detector
    args_dict = vars(opt)
    detect = Detect(**args_dict)

    workers_num = args_dict['workers']
    image_num = args_dict['img_num']
    batchsize = args_dict['bs']
    video_num = args_dict['videos']
    
    # the number of images processed by each worker
    imgs = image_num * video_num 
    
    # record all workers timestamp
    m = Manager()
    time_stamp = m.list()
    
    cmdQueues = [Queue()] * workers_num
    imgQueues = [Queue()] * workers_num
    
    detectors = [Process(target=runWorker, args=(Worker(i, image_num, batchsize, detect, cmdQueue=cmdQueues[i],
                                                        imgQueue=imgQueues[i],time_stamp=time_stamp),)) for i in range(workers_num)]
    controller = Process(target=runController, args=(Controller(detector_num=workers_num, img_num=imgs,
                                                                cmdQueues=cmdQueues, imgQueues=imgQueues),))
    
    for detector in detectors:
        detector.start()
    controller.start()

    for detector in detectors:
        detector.join()
    controller.join()
    
    duration = max(time_stamp) - min(time_stamp)
    print("***** The system end-to-end latency is : {:.3f}s *****".format(duration))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)