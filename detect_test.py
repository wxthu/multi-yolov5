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
    
    def __init__(self, detector_num=2, img_num=50, all_queue=None):
        self.act_id = 0  # active task id
        self.detector_num = detector_num
        self.img_num = img_num
        self.controller_state = {}  # 把所有detector的detector_state更新到自己的controller_state中
        for i in range(self.detector_num):
            self.controller_state.update({str(i): 'idle'})
        self.all_queue = all_queue
    
    def update_state_table(self, state_dict):
        """
        从detector接收到的state_dict, 更新到self.controller_state中去
        :param state_dict: 从detector接收到的detector_state
        """
        self.controller_state.update(state_dict)
    
    def init_msg(self):
        init_state = {}
        init_state.update({self.act_id: 'infer'})
        init_state.update({'img': []})
        return init_state
    
    def get_action(self):
        """
        根据当前的self.controller_state, 对所有的detector进行控制
        """
        new_state = {}
        print('active w : {}'.format(self.act_id))
        if self.controller_state[str(self.act_id)] == 'done':
            self.controller_state[str(self.act_id)] = 'idle'
            
            if self.act_id + 1 == self.detector_num:
                self.act_id = 0
            else:
                self.act_id += 1
            self.controller_state.update({str(self.act_id): 'infer'})
            new_state.update({str(self.act_id): 'infer'})
            print('control state update -> worker {}'.format(self.act_id))
            print('current cstate : {}'.format(self.controller_state))
        return new_state
    
    def run(self):
        for i in range(self.img_num):
            for q in self.all_queue:
                q.put(i)
    
    def run_backup(self):
        """
        UDP controller server. Accept msg from arbitrary address
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.bind(('127.0.0.1', 8000))
        client1_addr = ('127.0.0.1', 8001)  # activate one of client workers
        client2_addr = ('127.0.0.1', 8002)
        
        send_msg = self.init_msg()
        num = server_socket.sendto(encode_dict(send_msg), client1_addr)
        num = server_socket.sendto(encode_dict(send_msg), client2_addr)
        print('*** controller initial sending success {}***'.format(num))
        print("initial ctrl state : {} ".format(self.controller_state))
        interval = 1 / 500
        img_count = 1  # we have sent one img in init_msg function
        now = time.time()
        while True:
            # receive msg from arbitrary worker address
            recv_msg, client_addr = server_socket.recvfrom(115200)
            recv_msg = decode_dict(recv_msg)
            print("controller recv : {}".format(recv_msg))
            print("^^^ before update, ctrl state : {} ^^^".format(self.controller_state))
            self.update_state_table(recv_msg)
            print("^^^ update ctrl state : {} ^^^".format(self.controller_state))
            send_msg = self.get_action()
            print("&& after get_ac, ctrl state : {} &&".format(self.controller_state))
            print("to send to worker : {}".format(send_msg))
            
            if time.time() - now >= interval:
                if img_count < self.img_num:
                    send_msg.update({'img': []})
                    img_count += 1
                else:
                    print("control signal exit, send {} images".format(img_count))
                now = time.time()
            # send back to current worker address
            send_msg = encode_dict(send_msg)
            server_socket.sendto(send_msg, client1_addr)
            server_socket.sendto(send_msg, client2_addr)

class Worker:
    """
    单个detector
    """
    
    def __init__(self, index, img_num, batchsize, engine: Detect, queue, time_stamp):
        self.index = str(index)
        self.id = index
        self.img_num = img_num
        self.batchsize = batchsize
        self.engine = engine
        self.detector_state = {}  # detector_state记录了当前单个detector的各种信息
        self.detector_state.update({str(index): 'idle'})
        self.q = queue
        self.count = 0
        self.time_stamp = time_stamp
    
    def update_state(self):
        """
        更新self.detector_state
        """
        self.detector_state.update({self.index: 'done'})
        return
    
    def run(self):
        for i in range(self.img_num):
            self.q.get()
            if i == 0:
                self.time_stamp.append(time.time())
        
        print('worker{} get {} images'.format(self.id, self.img_num))
        
        self.time_stamp.append(time.time())
    
    def run_backup(self):
        sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sk.bind(('127.0.0.1', 8001+self.id))
        # server_addr = ('127.0.0.1', 8000)  
        finished = False
        while True:
            print('detector', self.index, 'img queue len is', len(self.q))
            recv_msg, server_addr = sk.recvfrom(115200)
            recv_msg = decode_dict(recv_msg)
            print('w{} receive : {}'.format(self.index, recv_msg))
            
            if 'img' in recv_msg:
                self.count += 1
                print("begin to append img")
                for _ in range(self.videos):
                    self.q.append(np.zeros(shape=(1920, 1080, 3)))
            else:
                print('worker recieve all the images : {}'.format(self.count * self.videos))
            
            if self.index in recv_msg and recv_msg[self.index] == 'infer':
                frames = []
                for _ in range(self.batchsize):
                    if len(self.q) > 0:
                        frames.append(self.q.pop())
                    else:
                        print('all the images have been processed by worker {}'.format(self.index))
                        finished = True
                        break
                if finished is True:
                    break
                pred = self.engine.run(np.stack(frames))
                self.update_state()
                print('current state : {}'.format(self.detector_state))
                
            send_msg = encode_dict(self.detector_state)
            sk.sendto(send_msg, server_addr)         

def detector_run(detector):
    detector.run()
    detector.run_backup()

def controller_run(c):
    c.run()

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
    parser.add_argument('--workers', type=int, default=2, help='the number of detector')
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
    
    # record all workers timestamp
    m = Manager()
    time_stamp = m.list()
    
    Queue_list = [Queue()] * workers_num
    
    detectors = [Process(target=detector_run, args=(Worker(i, image_num, batchsize, detect, queue=Queue_list[i], time_stamp=time_stamp),)) for i in range(workers_num)]
    controller = Process(target=controller_run, args=(Controller(detector_num=workers_num, img_num=image_num, all_queue=Queue_list),))
    
    for detector in detectors:
        detector.start()
    time.sleep(8) # to wait child process init
    controller.start()
    

    for detector in detectors:
        detector.join()
    controller.join()
    
 
    duration = max(time_stamp) - min(time_stamp)
    print("***** The system end-to-end latency is : {:.3f}s *****".format(duration))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)