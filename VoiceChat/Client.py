
import pyaudio
import socket
import pickle
import struct
from multiprocessing import Queue


class Client():
    """Socket client module"""
    def __init__(self, host, port):
        self.CONNECTED = False
        self.VID_RECV = True
        self.VOICE_RECV = False
        self.HEADER = 2048
        self.VOICE_HEADER = 4096
        self.CHUNKS = 4096
        self.ADDR = (host,port)
        self.FORMAT = "utf-8"
        
        self.Q = Queue(100) # Queue of receive image
        self.c = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
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

    def connect(self):
        """Connect to Server"""
        self.c.connect(self.ADDR)
        self.CONNECTED = True
        print("Connected")

    def send_msg(self, msg):
        """Send msg to Server"""
        message = bytes(msg, 'utf-8')
        self.c.sendall(message) 

    def recv_msg(self):
        """Receive message form Server"""
        receive = self.c.recv(self.HEADER).decode('utf-8')   # Receive message from server
        return str(receive)
    
    def disconnect(self):
        """Disconnect to Server"""
        self.VID_RECV = False
        self.c.close()

    def recv_video(self):
        """Receive video/frames from Server, add frames to the Queue"""
        data=b""
        size = struct.calcsize("Q")
        try:
            while True:
                while len(data) < size:
                    pack = self.c.recv(4*self.HEADER)
                    if not pack: 
                        break
                    data += pack
                packed_size = data[:size]
                data = data[size:]
                msg_size = struct.unpack("Q", packed_size)[0]
                while len(data)< msg_size:
                    data += self.c.recv(4*self.HEADER)
                image_data = data[:msg_size]
                data = data[msg_size:]
                image = pickle.loads(image_data)
                if not self.Q.full():
                    self.Q.put(image)
                if self.VID_RECV is False:
                    raise Exception("Recv Vid Completed")
        except Exception as e:
            return e 

    def send_voice(self):
        """Send voice to Server"""
        while True:
            data = self.input_audio.read(self.VOICE_HEADER)
            self.c.sendall(data)

    def recv_voice(self):
        """Receive voice form Server"""
        while True:
            voice_data = self.c.recv(self.VOICE_HEADER)
            self.output_audio.write(voice_data) 
            
if __name__ == '__main__':
    ip_test = "192.168.1.3"
    client = Client(ip_test, 6789)
    client.connect()
    #client.recv_voice() # GOOD
    client.send_voice()












