import socket
import os
from _thread import *

client_num = 8

ServerSocket = socket.socket()
host = '127.0.0.1'
port = 35491
ThreadCount = 0
try:
    ServerSocket.bind((host, port))
except socket.error as e:
    print(str(e))

print('Server start!')
ServerSocket.listen(client_num)


def threaded_client(connection):
    connection.send(str.encode('Welcome to the Server\n'))
    while True:
        data = connection.recv(2048)
        # TODO 拿到client的数据，做后续处理
        reply = 'Server Says: ' + data.decode('utf-8')
        print('got ',  data, ' from', connection)

        # TODO 返回处理完的结果
        connection.sendall(str.encode(reply))

        if not data:
            break

    connection.close()

while True:
    Client, address = ServerSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    start_new_thread(threaded_client, (Client, ))
    ThreadCount += 1
    print('Thread Number: ' + str(ThreadCount))
ServerSocket.close()