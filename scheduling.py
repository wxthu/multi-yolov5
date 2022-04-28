"""
We often have following configs for testing:
(i)  rnt50, rnt152, alexnet, vgg11, yolov5s, yolov5x 
(ii) sqt10, rnt18, rnt34, rnt101, yolov5s, yolov5x
(iii)sqt11, rnt50, rnt152, vgg13, yolo5s, yolov5x
"""

import argparse
import multiprocessing
from multiprocessing import Process, Queue, Manager
import os
import numpy as np
import time
import torch

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import alexnet, vgg16, vgg13, vgg11, squeezenet1_0, squeezenet1_1
from models.common import DetectMultiBackend
from utils.general import check_requirements, print_args
from utils.torch_utils import time_sync
from controller import Controller

class Detect:
    def __init__(self, model):
        self.model = model[1].eval()
        self.name = model[0]
        self.device = 'cpu'

    def load_model(self):
        self.device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.model.to(self.device)

    def release_model(self):
        self.device = 'cpu'
        self.model.to(self.device)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self, w_id, img):
        # Run inference
        dt = []
        t1 = time_sync()
        img = img.transpose((0, 3, 1, 2))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).type(torch.FloatTensor).to(self.device)
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        print('data shape is ', im.shape)  
        t2 = time_sync()
        dt.append(t2 - t1)
        
        # Inference
        self.model(im)
        t3 = time_sync()
        dt.append(t3 - t2)
        
        # Print results
        t = tuple(x * 1E3 for x in dt)
        print(f'Speed: %.1fms pre-process, %.1fms inference, worker id :%d' % (t + (w_id,)))

class Worker:
    
    def __init__(self, id, batchsize, c2wQueue, w2cQueue, imgQueue, time_stamp, model):
        self.id = id
        self.batchsize = batchsize
        self.engine = Detect(model)
        self.name = model[0]
        self.c2wQueue = c2wQueue
        self.w2cQueue = w2cQueue
        self.imgQueue = imgQueue
        self.time_stamp = time_stamp
        self.imgs = 0
        self.hasInfered = False # Set a flag to avoid the case where controller hasn't put img into imgQueue
        self.loaded = False     # To indicate whether model has loaded into GPU
    
    def run(self):
        self.time_stamp.append(time.time())

        while True:
            command = None if self.c2wQueue.empty() else self.c2wQueue.get()
            if command == 'to_load':
                self.engine.load_model()
                self.loaded = True
                    
            if command == 'to_infer':
                assert self.loaded == True
                self.hasInfered = True
                assert not self.imgQueue.empty()

                frames = []
                for _ in range(self.batchsize):
                    if self.imgQueue.qsize() > 0:
                        self.imgs = self.imgQueue.get()
                        frames.append(np.zeros(shape=(384, 640, 3)))

                assert len(frames) > 0

                pred = self.engine.run(self.id, np.stack(frames))
                # send signal to controller
                self.w2cQueue.put('batch_done')
                self.engine.release_model()
                self.loaded = False
                        
            # if self.imgQueue.empty() and self.hasInfered:
            #     print('all {} images have been processed by worker {}'.format(self.imgs, self.id))
            #     # release gpu memory
            #     self.engine.release_model()
            #     self.w2cQueue.put('finish')
            #     break
            if command == 'exit':
                print(f'worker {self.id} has processed {self.imgs} imgs and exit ...')
                break
                
        self.time_stamp.append(time.time())
        return              

def runWorker(i,
            batchsize, 
            c2wQueue,
            w2cQueue,
            imgQueue, 
            time_stamp,
            model):
    worker = Worker(i,
                    batchsize, 
                    c2wQueue=c2wQueue,
                    w2cQueue=w2cQueue,
                    imgQueue=imgQueue, 
                    time_stamp=time_stamp,
                    model=model
                    )
    worker.run()

def runController(workers_num, 
                  image_num,
                  video_num,
                  c2wQueues,
                  w2cQueues, 
                  imgQueues,
                  weights, 
                  prices,
                  total_memory):
    controller = Controller(detector_num=workers_num, img_num=image_num*video_num, c2wQueues=c2wQueues,
                            w2cQueues=w2cQueues, imgQueues=imgQueues, weights=weights, prices=prices,
                            mem=total_memory)
    controller.run()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1, help='batch size of img')
    parser.add_argument('--img_num', type=int, default=50, help='the number of img sent by each client')
    parser.add_argument('--videos', type=int, default=4, help='the number of video stream')
    parser.add_argument('--workers', type=int, default=6, help='the number of detector')
    parser.add_argument('--memory', type=int, default=1500, help='available gpu memory in the server')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    args_dict = vars(opt)

    # workers_num = args_dict['workers']
    workers_num = len(models)
    image_num = args_dict['img_num']
    batchsize = args_dict['bs']
    video_num = args_dict['videos']
    total_memory = args_dict['memory']
    
    # record all workers timestamp
    time_stamp = Manager().list()
    MAX_Q_NUM = 1000
    
    c2wQueues = [Queue(MAX_Q_NUM) for _ in range(workers_num)]
    w2cQueues = [Queue(MAX_Q_NUM) for _ in range(workers_num)]
    imgQueues = [Queue(MAX_Q_NUM) for _ in range(workers_num)]
    
    controller = Process(target=runController, args=(workers_num, 
                                                     image_num,
                                                     video_num,
                                                     c2wQueues,
                                                     w2cQueues,
                                                     imgQueues,
                                                     weights,
                                                     prices,
                                                     total_memory
                                                     )
                         )
    
    detectors = [
                Process(target=runWorker, 
                        args=(i,
                                batchsize,
                                c2wQueues[i],
                                w2cQueues[i],
                                imgQueues[i], 
                                time_stamp,
                                models[i]
                            )
                        )
                            
                for i in range(workers_num)
                ]

    controller.start()
    
    for detector in detectors:
        detector.start()

    controller.join()
    for detector in detectors:
        detector.join()
    
    warmup_overhead = 2.5
    duration = max(time_stamp) - min(time_stamp)
    print("***** The system end-to-end latency is : {:.3f}s *****".format(duration - warmup_overhead))


if __name__ == "__main__":
    models = []
    yolov5x = DetectMultiBackend('yolov5x.pt', device=torch.device('cpu'))
    yolov5s = DetectMultiBackend('yolov5s.pt', device=torch.device('cpu'))

    # profiling : memory, inference, total_time, gpu_utilization
    models.append(['sqt10', squeezenet1_0(), 58, {'infer_time':3.96, 'total_time':10.0, 'gpu_util':47}])
    models.append(['sqt11', squeezenet1_1(), 46, {'infer_time':3.52, 'total_time':9.62, 'gpu_util':40}])
    models.append(['rnt18', resnet18(), 124, {'infer_time':4.39, 'total_time':24.2, 'gpu_util':54}])
    models.append(['rnt34', resnet34(), 166, {'infer_time':7.54, 'total_time':43.22, 'gpu_util':54}])
    models.append(['rnt50', resnet50(), 192, {'infer_time':13.44, 'total_time':59.84, 'gpu_util':56}])
    models.append(['rnt101', resnet101(), 266, {'infer_time':22.75, 'total_time':132.62, 'gpu_util':50}])
    models.append(['rnt152', resnet152(), 334, {'infer_time':32.86, 'total_time':179.28, 'gpu_util':55}])
    models.append(['alexnet', alexnet(), 264, {'infer_time':4.24, 'total_time':136.29, 'gpu_util':39}])
    models.append(['vgg11', vgg11(), 618, {'infer_time':13.39, 'total_time':304.11, 'gpu_util':43}])
    models.append(['vgg13', vgg13(), 634, {'infer_time':18.59, 'total_time':306.89, 'gpu_util':44}])
    models.append(['vgg16', vgg16(), 674, {'infer_time':23.03, 'total_time':319.4, 'gpu_util':46}])
    models.append(['yolov5x', yolov5x, 422, {'infer_time':31.91, 'total_time':140.8, 'gpu_util':70}])
    models.append(['yolov5s', yolov5s, 60, {'infer_time':7.46, 'total_time':27.29, 'gpu_util':40}])

    weights = [x[2] for x in models]
    prices = [x[3]['gpu_util'] for x in models]

    opt = parse_opt()
    main(opt)