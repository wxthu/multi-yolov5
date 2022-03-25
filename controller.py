import socketserver
import time
from multiprocessing import Queue
from typing import Any
from encode_decode import encode_dict, decode_dict
import numpy as np


class Controller(socketserver.BaseRequestHandler):
    """
    作为Controller, 监视所有的detector
    """
    def __init__(self, request: Any, client_address: Any, server: socketserver.BaseServer, detector_num=3, img_num=50):
        self.act_id = 0  # active task id
        self.detector_num = detector_num
        self.img_num = img_num
        self.controller_state = {}   # 把所有detector的detector_state更新到自己的controller_state中
        for i in range(self.detector_num):
            self.controller_state.update({i :'ready'})
        # self.controller_state.update({self.act_id :'infer'})
        super().__init__(request, client_address, server)

    def update_state_table(self, state_dict):
        """
        从detector接收到的state_dict, 更新到self.controller_state中去
        :param state_dict: 从detector接收到的detector_state
        """
        self.controller_state.update(state_dict)

    def init_msg(self):
        init_state = {}
        init_state.update({self.act_id : 'infer'})
        init_state.update({'img': np.zeros(shape=(1920, 1080)).tolist()})
        return init_state
        
    def get_action(self):
        """
        根据当前的self.controller_state, 对所有的detector进行控制
        """
        new_state = {}
        if self.controller_state[self.act_id] == 'done':
            self.controller_state[self.act_id] = 'idle'
         
            if self.act_id + 1 == self.detector_num:
                self.act_id = 0
            else:
                self.act_id += 1
            new_state.update({self.act_id : 'infer'})
                
        return new_state

    def handle(self):
        conn = self.request
        interval = 1/3
        now = time.time()
        img_count = 1  # we have sent one img in init_msg function 
        
        send_dict = self.init_msg()
        conn.sendall(encode_dict(send_dict))
        while True:
            # TODO 接收detector的状态信息
            recv_dict = decode_dict(conn.recv(115200))
            
            # TODO 更新controller_state
            self.update_state_table(recv_dict)
            send_dict = self.get_action()

            if time.time() - now >= interval:
                if img_count < self.img_num:
                    send_dict.update({'img': np.zeros(shape=(1920, 1080)).tolist()})
                    img_count += 1
                now = time.time()
                
                conn.sendall(encode_dict(send_dict))


if __name__ == "__main__":
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 8010), Controller)
    server.serve_forever()
