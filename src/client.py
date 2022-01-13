import socket
from _thread import *
import time


def make_client(arg):

    socketObject = socket.socket()
    socketObject.connect(("localhost", 35491))
    print("Connected to localhost")
    
    # Send a message to the web server to supply a page as given by Host param of GET request
    HTTPMessage = "GET / HTTP/1.1\r\nHost: localhost\r\n Connection: close\r\n\r\n"
    bytes = str.encode(HTTPMessage)
    socketObject.sendall(bytes)
    
    # Receive the data
    while (True):
        time.sleep(2)
        # TODO 要发给server的数据data
        # data = input()
        socketObject.send(str.encode(arg))
        
        # TODO 从server收到的结果
        res = socketObject.recv(1024)
        
        print(res)
        if (arg == b''):
            print("Connection closed")
            break
    socketObject.close()
    
    
for i in range(5):
    start_new_thread(make_client, (str(i), ))
    
while True:
    stop_flag = input()
    if stop_flag == '':
        break