import json
import time
import random
import socket
import struct
import numpy as np
# import soundfile as sf
import sounddevice as sd
import audio

CHANNELS = 1
SAMPLE_RATE = 44100
BLOCK_SIZE = 64

def get_sample_data() -> np.ndarray:
    t = time.time()
    frequency = 440
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate)
    t = t[:64]
    buffer = np.sin(2 * np.pi * frequency * t) * 1
    
    # Add some noise (abs 1, max 1, min -1)
    buffer += np.random.random(64) * 0.1
    buffer = np.clip(buffer, -1, 1)

    return buffer

def stream(socket: socket.socket, device_index: int):
    # TODO(shoh): exception for not catching device because of audio paramaters
    input_stream = audio.get_intput_stream(
        device_index = device_index, 
        channels = CHANNELS, 
        sample_rate = SAMPLE_RATE, 
        blocksize = BLOCK_SIZE)

    input_stream.start()
    while True:
        # # Sample 
        # buffer = get_sample_data()

        # Device
        frame, overflow = input_stream.read(BLOCK_SIZE)
        np.clip(frame, -1, 1, out=frame)

        if overflow:
            pass
            # print("Overflow!")
        
        # Send the buffer
        socket.sendall(frame.astype(np.float32).tobytes())
        
        # # Wait a bit
        # time.sleep(0.01)

    input_stream.stop()

# Example client code for testing
def create_test_client():
    # Connect to server
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('127.0.0.1', 33333))
    
    # Register device
    random_id = random.randint(1, 1000)
    client.sendall("REGISTER:test_device_{}:Test Audio Device".format(str(random_id)).encode('utf-8'))
    response = client.recv(1024).decode('utf-8')
    print(f"Registration response: {response}")
    
    if response.startswith("REGISTERED"):
        # Send audio data
        print("Connection established, trying to transmit devices list...")

    device_list = json.dumps(audio.list_audio_devices())
    print(device_list)
    print(len(device_list))

    client.sendall(device_list.encode('utf-8'))

    response = client.recv(1024).decode('utf-8')
    print(f"Device list response: {response}")
    device_index = None
    if response.startswith("DEVICE_LIST_RECEIVED"):
        device_index = response.split(":")[1]
        print(f"Device index: {device_index}")

    device_index = int(device_index)
    try:
        stream(client, device_index)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()

# TODO(shoh): get mic data from devices
class Device:
    def __init__(self, name: str, id: str, mic_id: int, spk_id: int):
        self.name = name
        self.id = id
        self.mic_id = mic_id
        self.spk_id = spk_id

    def get_audio_data(self):
        pass

    def get_audio_data_from_mic(self):
        pass

    def get_audio_data_from_file(self):
        pass

device = Device('test_device', 'test_id', 0, 0)



if __name__ == "__main__":    
    # To test with a simulated client:
    import threading
    threading.Thread(target=create_test_client).start()