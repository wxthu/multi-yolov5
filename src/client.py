import socket
from _thread import *
import time

client_num = 8


# 创造一个新的client
def make_client(client_id):
    socketObject = socket.socket()
    socketObject.connect(("localhost", 35491))
    print("Connected to localhost")

    # 发送握手头信息
    HTTPMessage = "GET / HTTP/1.1\r\nHost: localhost\r\n Connection: close\r\n\r\n"
    socketObject.sendall(str.encode(HTTPMessage))

    # Receive the data
    while True:
        # TODO 设置间隔频率
        time.sleep(2)

        # TODO 要发给server的数据data
        socketObject.send(str.encode(client_id))

        # TODO 从server收到的结果
        res = socketObject.recv(1024)

        print(res)
        if res == b'':
            print("Connection closed")
            break
    socketObject.close()


for i in range(client_num):
    # 启动一个新的client
    start_new_thread(make_client, (str(i),))

while True:
    stop_flag = input()
    if stop_flag == '':
        break
