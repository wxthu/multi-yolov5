"""
"""

import argparse
import multiprocessing
from multiprocessing import Process, Queue, Manager
import os
import sys
import numpy as np
import time
import pandas as pd
import cv2
import torch

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import alexnet, vgg16, vgg13, vgg11, squeezenet1_0, squeezenet1_1
from models.common import DetectMultiBackend
from utils.general import check_requirements, print_args
from utils.torch_utils import time_sync
from decision import Decision


class Detect:
    def __init__(self, model):
        self.model = model.eval()
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
        im = torch.from_numpy(img).to(self.device)
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


class Controller:
    """
    作为Controller, 监视所有的Worker
    """
    def __init__(self, detector_num=3, img_num=250, c2wQueues=None, w2cQueues=None, imgQueues=None, capacity=3):
        """ 
        two-way communiaction to prevent message contention
        """
        self.cap = capacity  # the number of models accommodated by the device
        self.act_ids = [] 
        self.wait_ids = [] 
        self.img_num = img_num
        self.c2wQueues = c2wQueues
        self.w2cQueues = w2cQueues
        self.imgQueues = imgQueues
        self.exitSignal = 0  # when all workers finished job, exitSignal == detector_num
        self.cmds = []  # each element is bool list which indicate the models to be executed at the moment
        self.cursor = 0 # add cursor for cmds
        self.initialization()

    def initialization(self):
        for q in range(len(self.imgQueues)):
            for i in range(self.img_num):
                # Too large data will block process 
                # q.put(np.zeros(shape=(1920, 1080, 3)))
                self.imgQueues[q].put(i+1)

        df = pd.read_csv('mock_request_rate.csv')
        for col in df:
            self.cmds.append(df[col].values)

    def initQueues(self):
        cmd = self.cmds[self.cursor]
        candidate = list(range(len(cmd)))
        candidate = list(filter(lambda i : cmd[i] is True, candidate))
        self.cursor += 1

        det = Decision()
        select = det.decision()

        self.act_ids = [x - 1 for x in select]
        self.wait_ids = [x for x in candidate if x is not in self.act_ids]


    def popWorkers(self, lists):
        for e in lists:
            ind = self.act_ids.index(e)
            self.act_ids.pop(ind)

    def update_cmd_queue(self):
        popLists = []
        for j in range(len(self.act_ids)):
            if self.w2cQueues[self.act_ids[j]].empty() is False:
                cmd = self.w2cQueues[self.act_ids[j]].get()
                if cmd == 'batch_done':
                    self.wait_ids.append(self.act_ids[j])  # add into the tail

                if cmd == 'finish':
                    self.exitSignal += 1

                popLists.append(self.act_ids[j])

        self.popWorkers(popLists)
        while len(self.act_ids) < self.capacity:
            if len(self.wait_ids) > 0:
                new_id = self.wait_ids.pop(0)   # get new workers from head of queue
                self.c2wQueues[new_id].put('begin')
                self.c2wQueues[new_id].put('infer')  
                self.act_ids.append(new_id)
                print(f'add new worker {new_id} success !!!')
            else:
                print(f'no workers waiting for task...')
                break
            
        # print('GPU memory cannot hold more models temporarily ...')
        return

    def run(self):
        self.initQueues()
        while True:
            self.update_cmd_queue()
            if self.exitSignal >= self.detector_num:
                print("all workers has finished jobs !")
                break
        return
            

class Worker:

    def __init__(self, id, batchsize, c2wQueue, w2cQueue, imgQueue, time_stamp, model):
        self.id = id
        self.batchsize = batchsize
        self.engine = Detect(model[1])
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
            if self.c2wQueue.empty() is False and self.c2wQueue.get() == 'begin':
                self.engine.load_model()
                self.loaded = True
                    
            if self.loaded and self.c2wQueue.empty() is False and self.c2wQueue.get() == 'infer':
                self.hasInfered = True
                while self.imgQueue.empty() is False:
                    frames = []
                    for _ in range(self.batchsize):
                        if self.imgQueue.qsize() > 0:
                            self.imgs = self.imgQueue.get()
                            frames.append(np.zeros(shape=(384, 640, 3)))    
                    if len(frames) > 0:
                        pred = self.engine.run(self.id, np.stack(frames))
                        # send signal to controller
                        self.w2cQueue.put('batch_done')
                        self.engine.release_model()
                        self.loaded = False
                        break
                        
            if self.imgQueue.empty() and self.hasInfered:
                print('all {} images have been processed by worker {}'.format(self.imgs, self.id))
                # release gpu memory
                self.engine.release_model()
                self.w2cQueue.put('finish')
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
                    engine=model
                    )
    worker.run()

def runController(workers_num, 
                  image_num,
                  video_num,
                  c2wQueues,
                  w2cQueues, 
                  imgQueues,
                  capacity):
    controller = Controller(detector_num=workers_num, img_num=image_num*video_num, c2wQueues=c2wQueues,
                            w2cQueues=w2cQueues, imgQueues=imgQueues, capacity=capacity)
    controller.run()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1, help='batch size of img')
    parser.add_argument('--img_num', type=int, default=50, help='the number of img sent by each client')
    parser.add_argument('--videos', type=int, default=4, help='the number of video stream')
    parser.add_argument('--workers', type=int, default=6, help='the number of detector')
    parser.add_argument('--capacity', type=int, default=3, help='the number of supported detectors executing parallelly')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    models = []
    yolov5x = DetectMultiBackend('yolov5x.pt', device=torch.device('cpu'))
    yolov5s = DetectMultiBackend('yolov5s.pt', device=torch.device('cpu'))
    models.append(['sqt10', squeezenet1_0()])
    models.append(['sqt11', squeezenet1_1()])
    models.append(['rnt18', resnet18()])
    models.append(['rnt34', resnet34()])
    models.append(['rnt50', resnet50()])
    models.append(['rnt101', resnet101()])
    models.append(['rnt152', resnet152()])
    models.append(['alexnet', alexnet()])
    models.append(['vgg11', vgg11()])
    models.append(['vgg13', vgg13()])
    models.append(['vgg16', vgg16()])
    models.append(['yolov5x', yolov5x])
    models.append(['yolov5s', yolov5s])
    
    args_dict = vars(opt)

    # workers_num = args_dict['workers']
    workers_num = len(models)
    image_num = args_dict['img_num']
    batchsize = args_dict['bs']
    video_num = args_dict['videos']
    capacity = args_dict['capacity']
    
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
                                                     capacity,
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
    
    duration = max(time_stamp) - min(time_stamp)
    print("***** The system end-to-end latency is : {:.3f}s *****".format(duration))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)