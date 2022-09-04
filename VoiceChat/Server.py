
import socket
import pickle
import cv2 as cv
import struct
import pyaudio
from multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor
from time import sleep
class Server():
    def __init__(self,ip, port):
        self.FORMAT = "utf-8"
        self.HEADER = 2048
        self.VOICE_HEADER = 4096
        self.CHUNKS = 4096
        self.SEND_VID = True
        self.VOICE_RECV = False
        self.MUTE = True
        self.CONNECTED = False
        self.client_conn = None
        self.ip = ip
        self.port = port
        self.ip_port = (ip, port)

        self.VOICE_QUEUE = Queue(100)
        # PyAudio for voice data
        self.p = pyaudio.PyAudio()
        self.input_audio = self.p.open(input=True,
                                    format = pyaudio.paFloat32,
                                    rate=44100,
                                    channels=1,
                                    frames_per_buffer=self.CHUNKS)

        self.output_audio = self.p.open(output=True, 
                                format = pyaudio.paFloat32,
                                rate=44100,
                                channels=1,
                                frames_per_buffer=self.CHUNKS)

    def start(self):
        """Server is Listening"""
        self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sk .bind(self.ip_port)
        self.sk.listen(50)

    def connect(self):
        """Server is Waiting for connection"""
        print(f"[WAITING FOR CONNECTION AT {self.ip} -- {self.port}]")
        conn, addr = self.sk.accept()
        if conn:
            self.client_conn = conn
            self.CONNECTED =  True
            print("[CLIENT CONNECTED]:",addr)
            
    def close(self):
        """Close the connection of Client"""
        self.CONNECTED = False
        self.client_conn.close()
        self.client_conn = None
  
    def send_msg(self, msg):
        """Send a message to Client"""
        if self.client_conn is not None:
            message = bytes(msg, 'utf-8')
            self.client_conn.sendall(message) # Send message to server
    
    def recv_msg(self):
        """Receive message from Client"""
        if self.client_conn is not None:
            receive = self.client_conn.recv(self.HEADER).decode('utf-8') 
            return receive
    
    def send_video(self):
        """Send video to Client"""
        vid = cv.VideoCapture(0)
        vid.set(3, 320)
        vid.set(4,320)
        while self.client_conn is not None:
            if self.SEND_VID is True:
                ret, img = vid.read()
                if  ret:
                    try:
                        stream = pickle.dumps(img)
                        msg = struct.pack("Q", len(stream))+stream
                        self.client_conn.sendall(msg)
                    except Exception as e:
                        print(e)
        print('Send Vid End')

    def send_voice(self):
        """Server send voice to Client """
        while self.client_conn is not None:
            data = self.input_audio.read(self.VOICE_HEADER)
            self.client_conn.sendall(data)

    def recv_voice(self):
        """Server receive voice from Client"""
        while self.client_conn is not None:
            voice_data = self.client_conn.recv(self.VOICE_HEADER) # 4K
            self.output_audio.write(voice_data)


if __name__ =='__main__':
    ip_add = '192.168.1.3'
    server = Server(ip_add, 6789)
    server.start()
    server.connect()
    #server.send_voice() # GOOD
    server.recv_voice()

 