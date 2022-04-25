import pandas as pd

from decision import Decision

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
            
    def compare_w(self, i):
        return self.weights[i]
    
    def compare_p(self, i):
        return self.prices[i]
    
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
            candidate.sort(key=self.compare_p)
            
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
        # print(f"**** remkk;k;k; {reMem}")
        while reMem > 0:
            if len(self.wait_ids) > 0:
                # print(f'wait id : {self.wait_ids}')
                for ind in self.wait_ids:
                    # print(f'current wait : {ind}')
                    if self.weights[ind] <= reMem:
                        new_id = self.wait_ids.pop(0)   # get new workers from head of queue
                        self.c2wQueues[new_id].put('to_load')
                        self.c2wQueues[new_id].put('to_infer')
                        self.act_ids.append(new_id)
                        print(f'add new worker {new_id} success !!!')
                        reMem -= self.weights[new_id]
                        break
            else:
                # print(f'Unable to accommodate more models for the time being...')
                pass
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