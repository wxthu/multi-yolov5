import pickle
import socket
import sys
from _thread import *
import numpy as np
import struct
import time

from detect import Detect

payload_size = struct.calcsize(">L")

class Server:
    def __init__(self, detect, client_num=1, host='localhost', port=35490, buff_size=4096):
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
        res = self.detect.run(data)
        print('inference over', type(data), data.shape)
        return res

    def start_new_thread(self, connection):
        data = b""
        while True:
            while len(data) < payload_size:
                # print("Recv: {}".format(len(data)))
                data += connection.recv(self.buff_size)

            # print("Done Recv: {}".format(len(data)))
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            # print("msg_size: {}".format(msg_size))
            while len(data) < msg_size:
                data += connection.recv(self.buff_size)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            # 解码客户端传来的数据，数据为numpy数组
            # if sys.version_info.major < 3:
            #     decoded_data = pickle.loads(data)
            # else:
            #     decoded_data = pickle.loads(data, encoding='bytes')
            frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")

            t1 = time.time()
            result = self.process_np_array(frame)
            t2 = time.time()
            print('在外面的model inference 时间{:.3f}'.format(t2-t1))
            print()
            # connection.sendall(pickle.dumps('finished'))
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
