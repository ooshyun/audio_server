import json
from flask import Flask, render_template, request, jsonify
import socket
import threading
import numpy as np
import wave
import struct
import time
import os
from datetime import datetime
import scipy.signal as signal
import librosa

import sounddevice as sd

app = Flask(__name__)
EPS = 1e-9
class AudioServer:
    def __init__(self, host='0.0.0.0', port=33333):
        self.host = host
        self.port = port
        self.devices = {}  # Store registered devices
        self.is_recording = False
        self.out_stream = None
        self.out_stream_device = -1
        self.recorded_data = []

        # Initialize server socket
        self.init_socket()

        # Initialize audio data
        self.init_audio()

        # Initialize devices
        self.init_devices()

    def init_socket(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def init_audio(self):
        self.audio_config_json = json.load(open('conf/audio_config.json'))
        print(self.audio_config_json)
        self.buffer_size = self.audio_config_json['audio']['buffer_size']
        self.window_size = self.audio_config_json['audio']['window_size']
        self.hop_size = self.audio_config_json['audio']['hop_size']
        self.sample_rate = self.audio_config_json['audio']['sample_rate']
        # self.num_channel = 1
        self.current_fft_size = 0
        self.fft_size = self.audio_config_json['audio']['fft_size']
        self.fft_bin_size = self.audio_config_json['audio']['fft_size'] // 2 + 1
        self.overlap = self.audio_config_json['audio']['overlap']
        self.current_audio_data = np.zeros(2*self.window_size*self.sample_rate, 
                                           dtype=np.float32)
        self.num_frames = self.window_size * self.sample_rate // self.overlap + 1
        self.current_fft_frame = np.zeros(shape=(self.fft_bin_size,), 
                                          dtype=np.float32)
        self.current_fft_frame_amp = np.zeros(shape=(self.fft_bin_size,), 
                                              dtype=np.float32)
        self.current_stft = np.zeros(shape=(self.num_frames, self.fft_bin_size), 
                                            dtype=np.float32)

    def init_devices(self):
        self.device_list = {}
        self.spk_list = []
        self.current_device_id = 0

    def start_server(self):
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            print(f"Server started on {self.host}:{self.port}")
            
            # Start listener thread
            self.server_running = True
            self.listener_thread = threading.Thread(target=self.listen_for_connections)
            self.listener_thread.daemon = True
            self.listener_thread.start()
            
        except Exception as e:
            print(f"Error starting server: {e}")

    def stop_server(self):
        self.server_running = False
        for device_id, device_info in self.devices.items():
            try:
                device_info['socket'].close()
            except:
                pass
        try:
            self.server_socket.close()
        except:
            pass
        self.devices.clear()

        print("Server stopped")
        self.init_socket()
        self.init_audio()
        self.init_devices()

    def listen_for_connections(self):
        while self.server_running:
            try:
                client_socket, address = self.server_socket.accept()
                client_handler = threading.Thread(
                    target=self.handle_client_connection,
                    args=(client_socket, address)
                )
                client_handler.daemon = True
                client_handler.start()
            except:
                break

    def handle_client_connection(self, client_socket, address):
        try:
            data = client_socket.recv(1024).decode('utf-8')
            if data.startswith("REGISTER:"):
                parts = data.split(":", 2)
                if len(parts) >= 3:
                    device_id = parts[1]
                    device_name = parts[2]
                    self.devices[device_id] = {
                        'name': device_name,
                        'socket': client_socket,
                        'address': address,
                        'connected_time': datetime.now()
                    }
                    print(f"Device registered: {device_name} ({device_id}) from {address[0]}:{address[1]}")
                    client_socket.sendall("REGISTERED:OK".encode('utf-8'))

                    # Receive device list
                    self.current_device_id = device_id
                    data_device = client_socket.recv(4096).decode('utf-8')
                    data_device = json.loads(data_device)
                    self.device_list[device_id] = data_device
                    for key in self.device_list[device_id]:
                        self.device_list[device_id][key]["is_set"] = False

                    print(f"Received device list: {self.device_list[device_id]}")

                    device_index = self.send_device_index(device_id)
                    print(f"Device index: {device_index}")
                    client_socket.sendall(f"DEVICE_LIST_RECEIVED:{device_index}".encode('utf-8'))

                    self.receive_audio_data(device_id)
                else:
                    client_socket.sendall("ERROR:Invalid registration format".encode('utf-8'))
                    client_socket.close()
            else:
                client_socket.sendall("ERROR:Registration required".encode('utf-8'))
                client_socket.close()
        except Exception as e:
            print(f"Error handling connection: {e}")
            try:
                client_socket.close()
            except:
                pass

    def send_device_index(self, device_id):
        device_index = 0
        is_setup_device = False
        while not is_setup_device:
            print("Checking setting the index for device list...")
            for key in self.device_list[device_id]:
                if self.device_list[device_id][key]["is_set"] == True:
                    is_setup_device = True
                    device_index = key
            time.sleep(1)

        return device_index        

    def receive_audio_data(self, device_id):
        if device_id not in self.devices:
            return
        device = self.devices[device_id]
        client_socket = device['socket']
        while self.server_running:
            try:
                audio_data = client_socket.recv(self.buffer_size)
                if not audio_data:
                    break
                float_data = np.frombuffer(audio_data, dtype=np.float32)
                # TODO(shoh): benchmark time
                # if self.out_stream_device != -1:  # it works well in loopback, but in network, it has some delay
                #     self.process_out(float_data)
                # else:
                self.process_audio_data(float_data)
                self.process_fft_data(float_data)    
            except Exception as e:
                print(f"Error receiving audio data from {device['name']}: {e}")
                break
        print(f"Device disconnected: {device['name']} ({device_id})")
        try:
            client_socket.close()
        except:
            pass
        if device_id in self.devices:
            del self.devices[device_id]

    def process_audio_data(self, audio_data):
        print("Processing audio data...", len(audio_data), audio_data[:10])
        self.current_audio_data = np.roll(self.current_audio_data, -len(audio_data))
        self.current_audio_data[-len(audio_data):] = audio_data
        if self.is_recording:
            self.recorded_data.extend(audio_data)

    def process_fft_data(self, audio_data):
        self.current_fft_size += len(audio_data)
        if self.current_fft_size < self.fft_size:
            return
        else:
            if -self.current_fft_size+self.fft_size == 0:
                current_audio_frame = self.current_audio_data[-self.current_fft_size:]
            else:
                current_audio_frame = self.current_audio_data[-self.current_fft_size:-self.current_fft_size+self.fft_size]
            current_audio_frame = np.hanning(self.fft_size) * current_audio_frame
            current_fft_frame = np.fft.fft(current_audio_frame, n=self.fft_size)
            self.current_fft_frame_amp = np.abs(current_fft_frame[:self.fft_bin_size]) / self.fft_bin_size
            self.current_stft[:, :] = np.roll(self.current_stft, shift=-1, axis=0)
            self.current_stft[-1, :] = self.current_fft_frame_amp[:self.fft_bin_size]
            self.current_fft_size -= self.fft_size

    def process_out(self, audio_data):
        if self.out_stream is None:
            if self.out_stream_device == -1: return
            else:
                self.out_stream = sd.OutputStream(
                    device=self.out_stream_device,
                    channels=1,
                    samplerate=self.sample_rate,
                    dtype='float32'
                )
                self.out_stream.start()
        
        if self.out_stream:
            try:
                self.out_stream.write(audio_data)
            except Exception as e:
                print(f"Error writing to output stream: {e}")

    def get_audio_data__(self):
        return self.current_audio_data[self.window_size*self.sample_rate:]

    def get_fft_data__(self) -> np.ndarray:
        # return self.current_fft_frame_amp
        # normalized version
        return (self.current_fft_frame_amp - np.min(self.current_fft_frame_amp)) / (np.max(self.current_fft_frame_amp) - np.min(self.current_fft_frame_amp) + EPS)

    def get_stft_data__(self) -> np.ndarray:
        current_stft = self.current_stft.copy()
        for iframe in range(current_stft.shape[0]):
            range_stft = np.max(current_stft[iframe, :]) - np.min(current_stft[iframe, :])
            min_fft_bin = np.min(current_stft[iframe, :])
            current_stft[iframe, :] = (current_stft[iframe, :] - min_fft_bin) / (range_stft + EPS) * 255.
        return current_stft

    def get_device_list(self):
        if self.current_device_id == 0:
            return {}
        return self.device_list[self.current_device_id]

    def get_speaker_list(self):
        if len(self.spk_list) == 0:
            device_list = sd.query_devices()
            for i, device in enumerate(device_list):
                device.update({"is_set": False})
                self.spk_list.append(device)
        return self.spk_list

    def set_speaker_index(self, index):
        for idevice, device in enumerate(self.spk_list):
            if idevice == index:
                device["is_set"] = True
                self.out_stream_device = idevice
            else:
                device["is_set"] = False

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recorded_data = []
            print("Recording started")
        else:
            self.is_recording = False
            print(f"Recording stopped. Captured {len(self.recorded_data)} frames")

    def save_recording(self, filename):
        if not self.recorded_data:
            return "No recording data to save"
        try:
            data = np.array(self.recorded_data, dtype=np.float32)
            max_val = np.max(np.abs(data))
            if max_val > 1.0:
                data = data / max_val
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                data_int = (data * 32767).astype(np.int16)
                wf.writeframes(data_int.tobytes())
            return f"Recording saved to {filename}"
        except Exception as e:
            return f"Error saving recording: {e}"

audio_server = AudioServer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_server', methods=['POST'])
def start_server():
    audio_server.start_server()
    return jsonify({"status": "Server started"})

@app.route('/stop_server', methods=['POST'])
def stop_server():
    audio_server.stop_server()
    return jsonify({"status": "Server stopped"})

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    audio_server.toggle_recording()
    return jsonify({"status": "Recording toggled"})

@app.route('/save_recording', methods=['POST'])
def save_recording():
    filename = request.json['filename']
    result = audio_server.save_recording(filename)
    return jsonify({"status": result})

@app.route('/get_devices', methods=['GET'])
def get_devices():
    devices = [{"id": k, "name": v['name'], "address": v['address'][0]} for k, v in audio_server.devices.items()]
    return jsonify(devices)

@app.route('/get_microphones', methods=['GET'])
def get_microphones():
    # TODO(shoh): set device id (network connection)
    device_list = []
    # print(audio_server.get_device_list().keys())
    for device_id in audio_server.get_device_list().keys():
        device_list.append(audio_server.get_device_list()[device_id])
    # print(device_list)
    return jsonify(device_list)

@app.route('/get_speakers', methods=['GET'])
def get_speakers():
    return jsonify(audio_server.get_speaker_list())

@app.route('/get_audio_data', methods=['GET'])
def get_audio_data():
    return jsonify(audio_server.get_audio_data__().tolist())

@app.route('/get_fft_data', methods=['GET'])
def get_fft_data():
    return jsonify(audio_server.get_fft_data__().tolist())

@app.route('/get_stft_data', methods=['GET'])
def get_stft_data():
    return jsonify(audio_server.get_stft_data__().tolist())

@app.route('/get_audio_config', methods=['GET'])
def get_audio_config():
    return jsonify(audio_server.audio_config_json)

@app.route('/set_microphone_index', methods=['POST'])
def set_microphone_index():
    print("Setting device index...")
    device_id = request.json['device_id']
    audio_server.get_device_list()[device_id]["is_set"] = True
    return jsonify({"status": "Device index set"})

@app.route('/set_speaker_index', methods=['POST'])
def set_speaker_index():
    device_id = request.json['device_id']
    device_id = int(device_id)
    print("Setting speaker index...", device_id)
    audio_server.set_speaker_index(device_id)
    return jsonify({"status": "Speaker index set"})

if __name__ == '__main__':
    app.run(debug=True)
