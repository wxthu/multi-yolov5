import time
from encode_decode import encode_dict, decode_dict
import numpy as np
import socket


class Controller:
    """
    作为Controller, 监视所有的detector
    """
    
    def __init__(self, detector_num=1, img_num=10):
        self.act_id = 0  # active task id
        self.detector_num = detector_num
        self.img_num = img_num
        self.controller_state = {}  # 把所有detector的detector_state更新到自己的controller_state中
        for i in range(self.detector_num):
            self.controller_state.update({str(i): 'idle'})
    
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
            print('control state update -> worker {}'.format(self.act_id))
            print('current cstate : {}'.format(self.controller_state))
        return self.controller_state
    
    def run(self):
        """
        UDP controller server. Accept msg from arbitrary address
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.bind(('127.0.0.1', 8000))
        client1_addr = ('127.0.0.1', 8001) # activate one of client workers
        client2_addr = ('127.0.0.1', 8002)
        
        send_msg = self.init_msg()
        num = server_socket.sendto(encode_dict(send_msg), client1_addr)
        num = server_socket.sendto(encode_dict(send_msg), client2_addr)
        print('*** controller initial sending success {}***'.format(num))
        print("initial ctrl state : {} ".format(self.controller_state))
        interval = 1/50
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
            send_msgs = self.get_action()
            print("&& after get_ac, ctrl state : {} &&".format(self.controller_state))
            print("to send to worker : {}".format(send_msgs))
            
            if time.time() - now >= interval:
                if img_count < self.img_num:
                    send_msgs.update({'img' : []})
                    img_count += 1
                
                now = time.time()
            # send back to current worker address
            send_msgs = encode_dict(send_msgs)
            server_socket.sendto(send_msgs, client1_addr)
            server_socket.sendto(send_msgs, client2_addr)


if __name__ == "__main__":
    c = Controller()
    c.run()
