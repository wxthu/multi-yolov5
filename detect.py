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
import os
import sys
from pathlib import Path

from PIL.Image import Image
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox

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
    def __init__(self, img):
        self.img = img
        self.model = ''
        self.device = ''
        self.imgsz = 640

    def convertImage(self, stride=32, auto=True):
        img0 = cv2.imread(self.img)
        assert img0 is not None, f'Image Not Found'
        img = letterbox(img0, stride, auto)[0]
        
        img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img

    @torch.no_grad()
    def run(self,
            weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
            imgsz=(640, 640),  # inference size (height, width)
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn)
        stride, names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        half &= (pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if half else self.model.model.float()

        self.img = self.convertImage(stride=stride, auto=pt)

        # Run inference
        dt = []
        self.model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
        t1 = time_sync()
        im = torch.from_numpy(self.img).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()

        dt.append(t2 - t1)

        # Inference
        pred = self.model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt.append(t3 - t2)

        # Print results
        t = tuple(x * 1E3 for x in dt)  
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference' % t)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')   
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    # get input image
    input = ''

    detect = Detect(input)
    detect.run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
