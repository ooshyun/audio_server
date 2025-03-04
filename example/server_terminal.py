import socket
import threading
import numpy as np
import wave
import struct
import time
import os
from datetime import datetime
import scipy.signal as signal

class TerminalAudioServer:
    def __init__(self, host='0.0.0.0', port=33333):
        self.host = host
        self.port = port
        self.devices = {}  # Store registered devices
        self.current_audio_data = np.zeros(1024)  # Initial buffer for display
        self.is_recording = False
        self.recorded_data = []
        self.buffer_size = 64  # Buffer size as per requirement
        self.sample_rate = 16000  # Default sample rate
        self.server_running = False
        self.server_socket = None
        
        # Initialize server socket
        self.init_socket()

    def init_socket(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def log(self, message):
        """Print log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
    def start_server(self):
        """Start the server socket and listener thread"""
        if self.server_running:
            self.log("Server is already running")
            return
            
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.log(f"Server started on {self.host}:{self.port}")
            
            # Start listener thread
            self.server_running = True
            self.listener_thread = threading.Thread(target=self.listen_for_connections)
            self.listener_thread.daemon = True
            self.listener_thread.start()
            
            self.log("Server is now running. Press Ctrl+C to access the menu.")
            
        except Exception as e:
            self.log(f"Error starting server: {e}")
    
    def stop_server(self):
        """Stop the server and close all connections"""
        if not self.server_running:
            self.log("Server is not running")
            return
            
        self.server_running = False
        
        # Close all client connections
        for device_id, device_info in self.devices.items():
            try:
                device_info['socket'].close()
                self.log(f"Closed connection to {device_info['name']} ({device_id})")
            except:
                pass
        
        # Close server socket
        try:
            self.server_socket.close()
        except:
            pass
        
        self.devices.clear()
        self.log("Server stopped")

        self.init_socket()
    
    def listen_for_connections(self):
        """Listen for incoming device connections"""
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
                # Server socket closed or other error
                if self.server_running:
                    self.log("Error accepting connection")
                break
    
    def handle_client_connection(self, client_socket, address):
        """Handle device registration and communication"""
        try:
            # Device registration protocol
            # Expecting: "REGISTER:<device_id>:<device_name>"
            data = client_socket.recv(1024).decode('utf-8')
            if data.startswith("REGISTER:"):
                parts = data.split(":", 2)
                if len(parts) >= 3:
                    device_id = parts[1]
                    device_name = parts[2]
                    
                    # Register the device
                    self.devices[device_id] = {
                        'name': device_name,
                        'socket': client_socket,
                        'address': address,
                        'connected_time': datetime.now()
                    }
                    
                    self.log(f"Device registered: {device_name} ({device_id}) from {address[0]}:{address[1]}")
                    
                    # Send acknowledgment
                    client_socket.sendall("REGISTERED:OK".encode('utf-8'))
                    
                    # Start receiving audio data
                    self.receive_audio_data(device_id)
                else:
                    client_socket.sendall("ERROR:Invalid registration format".encode('utf-8'))
                    client_socket.close()
            else:
                client_socket.sendall("ERROR:Registration required".encode('utf-8'))
                client_socket.close()
        except Exception as e:
            self.log(f"Error handling connection: {e}")
            try:
                client_socket.close()
            except:
                pass
    
    def receive_audio_data(self, device_id):
        """Receive audio buffer from a device"""
        if device_id not in self.devices:
            return
        
        device = self.devices[device_id]
        client_socket = device['socket']
        
        while self.server_running:
            try:
                # Receive audio buffer (assuming fixed size of 64 as per requirement)
                audio_data = client_socket.recv(self.buffer_size * 4)  # 4 bytes per float
                
                if not audio_data:
                    # Connection closed
                    break
                
                # Convert bytes to float array
                float_data = np.frombuffer(audio_data, dtype=np.float32)
                
                # Process the audio data
                self.process_audio_data(float_data)
            except Exception as e:
                self.log(f"Error receiving audio data from {device['name']}: {e}")
                break
        
        # Cleanup on disconnect
        self.log(f"Device disconnected: {device['name']} ({device_id})")
        try:
            client_socket.close()
        except:
            pass
        
        # Remove from devices dict
        if device_id in self.devices:
            del self.devices[device_id]
    
    def process_audio_data(self, audio_data):
        """Process incoming audio data for visualization and recording"""
        # Update current audio data for visualization
        self.current_audio_data = np.roll(self.current_audio_data, -len(audio_data))
        self.current_audio_data[-len(audio_data):] = audio_data
        
        # If recording, store the audio data
        if self.is_recording:
            self.recorded_data.extend(audio_data)
    
    def start_recording(self):
        """Start recording audio data"""
        if self.is_recording:
            self.log("Recording is already in progress")
            return
            
        self.is_recording = True
        self.recorded_data = []
        self.log("Recording started")
    
    def stop_recording(self):
        """Stop recording audio data"""
        if not self.is_recording:
            self.log("No recording in progress")
            return
            
        self.is_recording = False
        self.log(f"Recording stopped. Captured {len(self.recorded_data)} samples")
    
    def save_recording(self, filename=None):
        """Save the recorded audio to a WAV file"""
        if not self.recorded_data:
            self.log("No recording data to save")
            return
        
        try:
            # Use provided filename or generate one
            if not filename:
                filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            # Ensure .wav extension
            if not filename.endswith('.wav'):
                filename += '.wav'
                
            # Convert data to numpy array
            data = np.array(self.recorded_data, dtype=np.float32)
            
            # Normalize to -1.0 to 1.0 if needed
            max_val = np.max(np.abs(data))
            if max_val > 1.0:
                data = data / max_val
            
            # Save as WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                
                # Convert float to PCM
                data_int = (data * 32767).astype(np.int16)
                wf.writeframes(data_int.tobytes())
            
            self.log(f"Recording saved to {filename}")
        except Exception as e:
            self.log(f"Error saving recording: {e}")
    
    def save_current_buffer(self, filename=None):
        """Save the current audio buffer to a WAV file"""
        if len(self.current_audio_data) == 0:
            self.log("No audio data in buffer")
            return
        
        try:
            # Use provided filename or generate one
            if not filename:
                filename = f"buffer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            # Ensure .wav extension
            if not filename.endswith('.wav'):
                filename += '.wav'
                
            # Create a copy of the data
            data = self.current_audio_data.copy()
            
            # Normalize to -1.0 to 1.0 if needed
            max_val = np.max(np.abs(data))
            if max_val > 1.0:
                data = data / max_val
            
            # Save as WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                
                # Convert float to PCM
                data_int = (data * 32767).astype(np.int16)
                wf.writeframes(data_int.tobytes())
            
            self.log(f"Current buffer saved to {filename}")
        except Exception as e:
            self.log(f"Error saving buffer: {e}")
    
    def list_connected_devices(self):
        """List all connected devices"""
        if not self.devices:
            self.log("No devices connected")
            return
        
        self.log(f"Connected devices ({len(self.devices)}):")
        for idx, (device_id, device_info) in enumerate(self.devices.items(), 1):
            connected_time = device_info['connected_time'].strftime("%Y-%m-%d %H:%M:%S")
            self.log(f"  {idx}. {device_info['name']} ({device_id}) - {device_info['address'][0]}:{device_info['address'][1]} - Connected since {connected_time}")
    
    def print_menu(self):
        """Print the main menu"""
        print("\n----- Audio Server Terminal Menu -----")
        print("1. Start server")
        print("2. Stop server")
        print("3. List connected devices")
        print("4. Start recording")
        print("5. Stop recording")
        print("6. Save recording")
        print("7. Save current audio buffer")
        print("8. Show stats")
        print("9. Exit")
        print("-------------------------------------")
    
    def show_stats(self):
        """Show current server statistics"""
        self.log("Server Statistics:")
        self.log(f"  Server running: {self.server_running}")
        self.log(f"  Connected devices: {len(self.devices)}")
        self.log(f"  Recording: {self.is_recording}")
        if self.is_recording:
            self.log(f"  Recorded samples: {len(self.recorded_data)}")
            duration = len(self.recorded_data) / self.sample_rate
            self.log(f"  Recording duration: {duration:.2f} seconds")
        self.log(f"  Current buffer size: {len(self.current_audio_data)} samples")
        
        # Calculate audio levels
        if len(self.current_audio_data) > 0:
            rms = np.sqrt(np.mean(np.square(self.current_audio_data)))
            peak = np.max(np.abs(self.current_audio_data))
            self.log(f"  Current audio RMS level: {rms:.6f}")
            self.log(f"  Current audio peak level: {peak:.6f}")
    
    def run(self):
        """Run the terminal audio server with an interactive menu"""
        print("Terminal Audio Server")
        print("Press Ctrl+C to access the menu at any time")
        
        running = True
        
        while running:
            try:
                # If the server is running, wait for keyboard interrupt
                if self.server_running:
                    try:
                        while self.server_running:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        pass
                
                # Show menu
                self.print_menu()
                choice = input("Enter your choice (1-9): ")
                
                if choice == '1':
                    self.start_server()
                elif choice == '2':
                    self.stop_server()
                elif choice == '3':
                    self.list_connected_devices()
                elif choice == '4':
                    self.start_recording()
                elif choice == '5':
                    self.stop_recording()
                elif choice == '6':
                    filename = input("Enter filename (leave empty for auto-generated name): ")
                    self.save_recording(filename if filename else None)
                elif choice == '7':
                    filename = input("Enter filename (leave empty for auto-generated name): ")
                    self.save_current_buffer(filename if filename else None)
                elif choice == '8':
                    self.show_stats()
                elif choice == '9':
                    if self.server_running:
                        confirm = input("Server is still running. Are you sure you want to exit? (y/n): ")
                        if confirm.lower() == 'y':
                            self.stop_server()
                            running = False
                    else:
                        running = False
                else:
                    self.log("Invalid choice, please try again")
                    
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected")
            except Exception as e:
                self.log(f"Error: {e}")
        
        self.log("Exiting Terminal Audio Server")

if __name__ == "__main__":
    server = TerminalAudioServer()
    server.run()