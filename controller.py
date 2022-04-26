import pandas as pd

from decision import Decision

class Controller:
    """
    作为Controller, 监视所有的Worker
    """
    def __init__(self, detector_num=3, img_num=800, mem=1024, config=None, strategy='infer_time',
                c2wQueues=None, w2cQueues=None, imgQueues=None):
        """ 
        two-way communiaction to prevent message contention
        """
        self.weights = []
        self.prices = []
        self.batches = []
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
        self.config = config  # list, each element is dictionary
        self.strat = strategy
        
    def initialization(self):
        for q in range(len(self.imgQueues)):
            for i in range(self.img_num):
                # Too large data will block process 
                # q.put(np.zeros(shape=(1920, 1080, 3)))
                self.imgQueues[q].put(i+1)

        df = pd.read_csv('mock_batched_request_rate.csv')
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
            wts = [0] 
            prc = [0]
            candidate = []
            for i, batch in enumerate(cmd):
                if batch:
                    candidate.append(i)
                    self.batches.append(batch)
                    self.weights.append(self.config[i][batch][0])
                    self.prices.append(self.config[i][batch][1][self.strat])
                    wts.append(self.config[i][batch][0])
                    prc.append(self.config[i][batch][1][self.strat])
                else:
                    # when detector i is not selected , we still add one element in the list 
                    # to ensure the consistency of indexing
                    self.weights.append(self.config[i][1][0])
                    self.prices.append(self.config[i][1][1][self.strat])
            
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

                # here we treat list as queue and pop the head element
                assert len(self.batches) > 0
                self.c2wQueues[self.act_ids[i]].put(self.batches.pop(0))
        
        popLists = []
        for j in range(len(self.act_ids)):
            if self.w2cQueues[self.act_ids[j]].empty() is False:
                cmd = self.w2cQueues[self.act_ids[j]].get()
                if cmd == 'batch_done':
                    self.task_num -= 1

                popLists.append(self.act_ids[j])

        self.popWorkers(popLists)
        popLists.clear()
        reMem = self.remainingMemory()

        while reMem > 0:
            if len(self.wait_ids) > 0:
                for ind in self.wait_ids:
                    if self.weights[ind] <= reMem:
                        popLists.append(self.wait_ids.index(ind))
                        self.c2wQueues[ind].put('to_load')
                        self.c2wQueues[ind].put('to_infer')

                        assert len(self.batches) > 0
                        self.c2wQueues[ind].put(self.batches.pop(0))
                        self.act_ids.append(ind)
                        print(f'add new worker {ind} success !!!')
                        reMem -= self.weights[ind]
                        break
            else:
                # print(f'Unable to accommodate more models for the time being...')
                pass
            break
        
        self.popWorkers(popLists)
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