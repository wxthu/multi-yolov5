import socket
from _thread import *
import time
import numpy as np
import pickle
import os


class Clients:
    def __init__(self, client_num=8, host='localhost', port=35491, freq=1.0, buff_size=4096):
        self.host = host
        self.client_num = client_num
        self.port = port
        self.freq = freq   # 每个线程每隔freq秒发送一次图片
        self.buff_size = buff_size

    def run(self):
        for i in range(self.client_num):
            # 启动一个新的client
            start_new_thread(self.single_client_thread, (i,))

    def send_np_array(self, handler, data):
        serialized_data = pickle.dumps(data, protocol=2)
        handler.sendall(serialized_data)

    def recv_np_array(self, handler):
        res = b''
        res += handler.recv(self.buff_size)
        res_data = pickle.loads(res, encoding='bytes')
        return res_data

    def single_client_thread(self, client_id:int):
        # 连接至服务器
        socket_handler = socket.socket()
        socket_handler.connect((self.host, self.port))
        print("Connected to localhost")

        # 不断地在loop内发送数据到服务器
        while True:
            time.sleep(self.freq)

            # TODO 要发给server的数据data
            data = np.ones((client_id + 1,))*client_id
            self.send_np_array(socket_handler, data)

            # TODO 从server收到的结果
            res_data = self.recv_np_array(socket_handler)

            print('client', client_id, '收到的数据', res_data, '\n')
            if type(res_data) is not np.ndarray:
                print("Connection closed")
                break
        socket_handler.close()


C = Clients()
C.run()

# 等待所有线程退出
while True:
    if input() == '':
        break
