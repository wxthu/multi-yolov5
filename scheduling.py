"""
"""

import argparse
import multiprocessing
from multiprocessing import Process, Queue, Manager
import os
import sys
import numpy as np
import time

import cv2
import torch

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import alexnet, vgg16, vgg13, vgg11, squeezenet1_0, squeezenet1_1
from models.common import DetectMultiBackend
from utils.general import check_requirements, print_args
from utils.torch_utils import time_sync
from decision import Decision
import pandas as pd


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
                print(f'worker {self.id} exit ...')
                break
                
        self.time_stamp.append(time.time())
        return              

class Controller:
    """
    作为Controller, 监视所有的Worker
    """
    def __init__(self, detector_num=3, img_num=250, c2wQueues=None, w2cQueues=None, imgQueues=None, 
                weights=None, prices=None, mem=1024):
        """ 
        two-way communiaction to prevent message contention
        """
        self.weights = weights
        self.prices = prices
        self.memory = mem  # memory contraint
        self.detector_num = detector_num
        self.act_ids = [] 
        self.wait_ids = [] 
        self.img_num = img_num
        self.c2wQueues = c2wQueues
        self.w2cQueues = w2cQueues
        self.imgQueues = imgQueues
        self.exitSignal = 0  # when all workers finished job, exitSignal == detector_num
        self.cmds = []  # each element is bool list which indicate the models to be executed at the moment
        self.cursor = 0 # add cursor for cmds
        self.task_num = 0 # record the number of tasks at the moment
        self.total_requests = 0
        
    def initialization(self):
        for q in range(len(self.imgQueues)):
            for i in range(self.img_num):
                # Too large data will block process 
                # q.put(np.zeros(shape=(1920, 1080, 3)))
                self.imgQueues[q].put(i+1)

        df = pd.read_csv('mock_request_rate.csv')
        for col in df:
            self.cmds.append(df[col].values)

        for x in self.cmds:
            self.total_requests += sum(x)
            
    def compare(self, i):
        return self.weights[i]
    
    def remainingMemory(self):
        total = 0
        for i in self.act_ids:
            total += self.weights[i]
        
        return self.memory - total
    
    def update_cmd_queue(self):
        if self.task_num == 0:
            cmd = self.cmds[self.cursor]

            # add 0 in the front of the list for dynamic programming
            wts = [0] + [self.weights[i] for i, elem in enumerate(cmd) if elem]
            prc = [0] + [self.prices[i] for i, elem in enumerate(cmd) if elem]
            candidate = [i for i, elem in enumerate(cmd) if elem]
            candidate.sort(key=self.compare)
            
            self.task_num = len(candidate)
            self.cursor += 1

            det = Decision(weights=wts, prices=prc, number=len(candidate), capacity=self.memory)
            select = det.decision()

            self.act_ids = [candidate[x - 1] for x in select]

            assert all(x in candidate for x in self.act_ids)
            self.wait_ids = [x for x in candidate if x not in self.act_ids]

            # To notify worker to load model into GPU memory
            for i in range(len(self.act_ids)):
                self.c2wQueues[self.act_ids[i]].put('to_load')
                self.c2wQueues[self.act_ids[i]].put('to_infer')
        
        popLists = []
        for j in range(len(self.act_ids)):
            if self.w2cQueues[self.act_ids[j]].empty() is False:
                cmd = self.w2cQueues[self.act_ids[j]].get()
                if cmd == 'batch_done':
                    self.task_num -= 1

                popLists.append(self.act_ids[j])

        self.popWorkers(popLists)
        reMem = self.remainingMemory()
        while reMem > 0:
            if len(self.wait_ids) > 0 and self.weights[self.wait_ids[0]] <= reMem:
                new_id = self.wait_ids.pop(0)   # get new workers from head of queue
                self.c2wQueues[new_id].put('to_load')
                self.c2wQueues[new_id].put('to_infer')
                self.act_ids.append(new_id)
                print(f'add new worker {new_id} success !!!')
                reMem -= weights[new_id]
            else:
                # print(f'Unable to accommodate more models for the time being...')
                break

        return

    def popWorkers(self, lists):
        for e in lists:
            ind = self.act_ids.index(e)
            self.act_ids.pop(ind)

    def run(self):
        self.initialization()
        while self.cursor < len(self.cmds):
            self.update_cmd_queue()

        for i in range(self.detector_num):
            self.c2wQueues[i].put('exit')
        
        print(f"all workers has finished jobs, processing {self.total_requests} images in total!")
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
    parser.add_argument('--capacity', type=int, default=3, help='the number of supported detectors executing parallelly')
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
    
    duration = max(time_stamp) - min(time_stamp)
    print("***** The system end-to-end latency is : {:.3f}s *****".format(duration))


if __name__ == "__main__":
    models = []
    yolov5x = DetectMultiBackend('yolov5x.pt', device=torch.device('cpu'))
    yolov5s = DetectMultiBackend('yolov5s.pt', device=torch.device('cpu'))
    # models.append(['sqt10', squeezenet1_0(), 38, 3.96])
    # models.append(['sqt11', squeezenet1_1(), 26, 3.52])
    # models.append(['rnt18', resnet18(), 104, 4.39])
    # models.append(['rnt34', resnet34(), 146, 7.54])
    models.append(['rnt50', resnet50(), 172, 13.44])
    # models.append(['rnt101', resnet101(), 246, 22.75])
    models.append(['rnt152', resnet152(), 318, 32.86])
    models.append(['alexnet', alexnet(), 246, 4.24])
    models.append(['vgg11', vgg11(), 598, 13.39])
    # models.append(['vgg13', vgg13(), 614, 18.59])
    # models.append(['vgg16', vgg16(), 654, 23.03])
    models.append(['yolov5x', yolov5x, 410, 31.91])
    models.append(['yolov5s', yolov5s, 40, 7.46])

    weights = [x[2] for x in models]
    prices = [x[3] for x in models]
    total_memory = 1024

    opt = parse_opt()
    main(opt)