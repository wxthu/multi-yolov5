import time
from encode_decode import encode_dict, decode_dict
import numpy as np
import socket


class Controller:
    """
    作为Controller, 监视所有的detector
    """
    
    def __init__(self, detector_num=2, img_num=50):
        self.act_id = 0  # active task id
        self.detector_num = detector_num
        self.img_num = img_num
        self.controller_state = {}  # 把所有detector的detector_state更新到自己的controller_state中
        for i in range(self.detector_num):
            self.controller_state.update({i: 'ready'})
    
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
        if self.controller_state[str(self.act_id)] == 'done':
            self.controller_state[str(self.act_id)] = 'idle'
            
            if self.act_id + 1 == self.detector_num:
                self.act_id = 0
            else:
                self.act_id += 1
            new_state.update({str(self.act_id): 'infer'})
        
        return new_state
    
    def run(self):
        """
        UDP controller server. Accept msg from arbitrary address
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.bind(('127.0.0.1', 8011))

        while True:
            # receive msg from arbitrary worker address
            recv_msg, client_addr = server_socket.recvfrom(115200)
            recv_msg = decode_dict(recv_msg)
            print(recv_msg)

            # send back to current worker address
            send_msg = encode_dict(recv_msg)
            server_socket.sendto(send_msg, client_addr)


if __name__ == "__main__":
    c = Controller()
    c.run()
