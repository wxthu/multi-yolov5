import pickle
import socket
import sys
from _thread import *
import numpy as np

from detect import Detect

class Server:
    def __init__(self, client_num=8, host='localhost', port=35491, buff_size=4096):
        self.host = host
        self.client_num = client_num
        self.port = port
        self.buff_size = buff_size

        self.server_handler = socket.socket()
        self.server_handler.bind((host, port))
        self.server_handler.listen(client_num)

        self.ThreadCount = 0

    def process_np_array(self, inference:Detect, data:cv2.imread):
        # TODO 在这里添加处理数据逻辑
        # 例如 result = data+1
        inference.run(data)
        return data

    def start_new_thread(self, connection):
        while True:
            data = b''
            data += connection.recv(self.buff_size)

            if not data:
                break

            # 解码客户端传来的数据，数据为numpy数组
            if sys.version_info.major < 3:
                decoded_data = pickle.loads(data)
            else:
                decoded_data = pickle.loads(data, encoding='bytes')

            result = self.process_np_array(decoded_data)
            connection.sendall(pickle.dumps(result, protocol=2))
        print('end thread connection')
        connection.close()

    def run(self):
        while True:
            client, addr = self.server_handler.accept()
            print('Connected to: ' + addr[0] + ':' + str(addr[1]))

            start_new_thread(self.start_new_thread, (client,))
            self.ThreadCount += 1
            print('Thread Number: ' + str(self.ThreadCount))

    def end(self):
        self.server_handler.close()


s = Server()
s.run()
s.end()
