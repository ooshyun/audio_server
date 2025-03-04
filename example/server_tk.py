import socket
import threading
import numpy as np
import wave
import struct
import time
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import scipy.signal as signal

class AudioServer:
    def __init__(self, host='0.0.0.0', port=33333):
        self.host = host
        self.port = port
        self.devices = {}  # Store registered devices
        self.current_audio_data = np.zeros(1024)  # Initial buffer for display
        self.is_recording = False
        self.recorded_data = []
        self.buffer_size = 64  # Buffer size as per requirement
        self.sample_rate = 16000  # Default sample rate
        
        # Setup GUI
        self.setup_gui()

        # Initialize server socket
        self.init_socket()

    def init_socket(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Audio Server")
        self.root.geometry("1200x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.main_tab = ttk.Frame(self.notebook)
        self.spectrogram_tab = ttk.Frame(self.notebook)
        self.waveform_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.main_tab, text="Main")
        self.notebook.add(self.spectrogram_tab, text="Spectrogram")
        self.notebook.add(self.waveform_tab, text="Waveform")
        
        # Main tab content
        self.setup_main_tab()
        
        # Spectrogram tab content
        self.setup_spectrogram_tab()
        
        # Waveform tab content
        self.setup_waveform_tab()
        
    def setup_main_tab(self):
        # Server control frame
        control_frame = ttk.LabelFrame(self.main_tab, text="Server Control")
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Start/Stop server button
        self.server_button = ttk.Button(control_frame, text="Start Server", command=self.toggle_server)
        self.server_button.pack(side='left', padx=10, pady=10)
        
        # Server status
        self.status_label = ttk.Label(control_frame, text="Server Status: Stopped")
        self.status_label.pack(side='left', padx=10, pady=10)
        
        # Recording control frame
        recording_frame = ttk.LabelFrame(self.main_tab, text="Recording Control")
        recording_frame.pack(fill='x', padx=10, pady=10)
        
        # Start/Stop recording button
        self.record_button = ttk.Button(recording_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side='left', padx=10, pady=10)
        self.record_button.config(state='disabled')
        
        # Save recording button
        self.save_button = ttk.Button(recording_frame, text="Save Recording", command=self.save_recording)
        self.save_button.pack(side='left', padx=10, pady=10)
        self.save_button.config(state='disabled')
        
        # Connected devices frame
        devices_frame = ttk.LabelFrame(self.main_tab, text="Connected Devices")
        devices_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Devices listbox
        self.devices_listbox = tk.Listbox(devices_frame)
        self.devices_listbox.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Log frame
        log_frame = ttk.LabelFrame(self.main_tab, text="Server Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Log text widget
        self.log_text = tk.Text(log_frame, height=10)
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
        
    def setup_spectrogram_tab(self):
        # Control frame
        control_frame = ttk.Frame(self.spectrogram_tab)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Save button
        self.save_spec_button = ttk.Button(control_frame, text="Save Spectrogram Data", 
                                          command=lambda: self.save_visualization_data('spectrogram'))
        self.save_spec_button.pack(side='left', padx=10, pady=10)
        
        # Spectrogram figure
        self.spec_fig = Figure(figsize=(10, 6))
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_canvas = FigureCanvasTkAgg(self.spec_fig, master=self.spectrogram_tab)
        self.spec_canvas.draw()
        self.spec_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Setup draggable functionality
        self.spec_canvas.mpl_connect('button_press_event', self.on_spectrogram_click)
        self.spec_canvas.mpl_connect('button_release_event', self.on_spectrogram_release)
        self.spec_selection_start = None
        self.spec_selection_end = None
        
    def setup_waveform_tab(self):
        # Control frame
        control_frame = ttk.Frame(self.waveform_tab)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Save button
        self.save_wave_button = ttk.Button(control_frame, text="Save Waveform Data", 
                                          command=lambda: self.save_visualization_data('waveform'))
        self.save_wave_button.pack(side='left', padx=10, pady=10)
        
        # Waveform figure
        self.wave_fig = Figure(figsize=(10, 6))
        self.wave_ax = self.wave_fig.add_subplot(111)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, master=self.waveform_tab)
        self.wave_canvas.draw()
        self.wave_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Setup draggable functionality
        self.wave_canvas.mpl_connect('button_press_event', self.on_waveform_click)
        self.wave_canvas.mpl_connect('button_release_event', self.on_waveform_release)
        self.wave_selection_start = None
        self.wave_selection_end = None
        
    def log(self, message):
        """Add message to log with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        print(log_message, end='')
        
    def toggle_server(self):
        """Start or stop the server"""
        if self.server_button['text'] == "Start Server":
            # Start server
            self.start_server()
            self.server_button['text'] = "Stop Server"
            self.status_label['text'] = "Server Status: Running"
            self.record_button.config(state='normal')
        else:
            # Stop server
            self.stop_server()
            self.server_button['text'] = "Start Server"
            self.status_label['text'] = "Server Status: Stopped"
            self.record_button.config(state='disabled')
            self.save_button.config(state='disabled')
    
    def toggle_recording(self):
        """Start or stop recording"""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.recorded_data = []
            self.record_button['text'] = "Stop Recording"
            self.save_button.config(state='disabled')
            self.log("Recording started")
        else:
            # Stop recording
            self.is_recording = False
            self.record_button['text'] = "Start Recording"
            self.save_button.config(state='normal')
            self.log(f"Recording stopped. Captured {len(self.recorded_data)} frames")
    
    def start_server(self):
        """Start the server socket and listener thread"""
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.log(f"Server started on {self.host}:{self.port}")
            
            # Start listener thread
            self.server_running = True
            self.listener_thread = threading.Thread(target=self.listen_for_connections)
            self.listener_thread.daemon = True
            self.listener_thread.start()
            
            # Start visualization update thread
            self.visualization_thread = threading.Thread(target=self.update_visualizations)
            self.visualization_thread.daemon = True
            self.visualization_thread.start()
            
        except Exception as e:
            self.log(f"Error starting server: {e}")
    
    def stop_server(self):
        """Stop the server and close all connections"""
        self.server_running = False
        
        # Close all client connections
        for device_id, device_info in self.devices.items():
            try:
                device_info['socket'].close()
            except:
                pass
        
        # Close server socket
        try:
            self.server_socket.close()
        except:
            pass
        
        self.devices.clear()
        self.devices_listbox.delete(0, tk.END)
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
                    self.root.after(0, lambda: self.update_devices_list())
                    
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
            self.root.after(0, lambda: self.update_devices_list())
    
    def process_audio_data(self, audio_data):
        """Process incoming audio data for visualization and recording"""
        # Update current audio data for visualization
        self.current_audio_data = np.roll(self.current_audio_data, -len(audio_data))
        self.current_audio_data[-len(audio_data):] = audio_data
        
        # If recording, store the audio data
        if self.is_recording:
            self.recorded_data.extend(audio_data)
    
    def update_visualizations(self):
        """Update the spectrogram and waveform visualizations"""
        while self.server_running:
            # Update waveform
            self.update_waveform()
            
            # Update spectrogram
            self.update_spectrogram()
            
            # Wait a short time for next update
            time.sleep(0.1)
    
    def update_waveform(self):
        """Update the waveform visualization"""
        try:
            # Get a copy of the current audio data
            audio_data = self.current_audio_data.copy()
            
            # Update plot on the main thread
            self.root.after(0, lambda: self._update_waveform_plot(audio_data))
        except Exception as e:
            self.log(f"Error updating waveform: {e}")
    
    def _update_waveform_plot(self, audio_data):
        """Update the waveform plot (called on the main thread)"""
        try:
            self.wave_ax.clear()
            self.wave_ax.plot(audio_data)
            self.wave_ax.set_title("Real-time Waveform")
            self.wave_ax.set_xlabel("Sample")
            self.wave_ax.set_ylabel("Amplitude")
            
            # Draw selection if active
            if self.wave_selection_start is not None and self.wave_selection_end is not None:
                self.wave_ax.axvspan(self.wave_selection_start, self.wave_selection_end, 
                                    color='red', alpha=0.3)
            
            self.wave_canvas.draw()
        except Exception as e:
            self.log(f"Error rendering waveform: {e}")
    
    def update_spectrogram(self):
        """Update the spectrogram visualization"""
        try:
            # Get a copy of the current audio data
            audio_data = self.current_audio_data.copy()
            
            # Compute spectrogram
            f, t, Sxx = signal.spectrogram(audio_data, fs=self.sample_rate, 
                                          window='hann', nperseg=256, noverlap=128)
            
            # Update plot on the main thread
            self.root.after(0, lambda: self._update_spectrogram_plot(f, t, Sxx))
        except Exception as e:
            self.log(f"Error computing spectrogram: {e}")
    
    def _update_spectrogram_plot(self, f, t, Sxx):
        """Update the spectrogram plot (called on the main thread)"""
        try:
            self.spec_ax.clear()
            self.spec_ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
            self.spec_ax.set_title("Real-time Spectrogram")
            self.spec_ax.set_ylabel('Frequency [Hz]')
            self.spec_ax.set_xlabel('Time [sec]')
            
            # Draw selection if active
            if self.spec_selection_start is not None and self.spec_selection_end is not None:
                self.spec_ax.axvspan(self.spec_selection_start, self.spec_selection_end, 
                                    color='red', alpha=0.3)
            
            self.spec_canvas.draw()
        except Exception as e:
            self.log(f"Error rendering spectrogram: {e}")
    
    def update_devices_list(self):
        """Update the devices listbox"""
        self.devices_listbox.delete(0, tk.END)
        for device_id, device_info in self.devices.items():
            self.devices_listbox.insert(tk.END, 
                                       f"{device_info['name']} ({device_id}) - {device_info['address'][0]}:{device_info['address'][1]}")
    
    def save_recording(self):
        """Save the recorded audio to a WAV file"""
        if not self.recorded_data:
            self.log("No recording data to save")
            return
        
        try:
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )
            
            if not filename:
                return
            
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
    
    def on_spectrogram_click(self, event):
        """Handle click on spectrogram for selection"""
        if event.inaxes == self.spec_ax:
            self.spec_selection_start = event.xdata
            self.spec_selection_end = None
    
    def on_spectrogram_release(self, event):
        """Handle release on spectrogram for selection"""
        if event.inaxes == self.spec_ax and self.spec_selection_start is not None:
            self.spec_selection_end = event.xdata
            # Ensure start < end
            if self.spec_selection_start > self.spec_selection_end:
                self.spec_selection_start, self.spec_selection_end = self.spec_selection_end, self.spec_selection_start
            
            # Update visualization
            self._update_spectrogram_plot(*signal.spectrogram(
                self.current_audio_data, fs=self.sample_rate, 
                window='hann', nperseg=256, noverlap=128
            ))
    
    def on_waveform_click(self, event):
        """Handle click on waveform for selection"""
        if event.inaxes == self.wave_ax:
            self.wave_selection_start = event.xdata
            self.wave_selection_end = None
    
    def on_waveform_release(self, event):
        """Handle release on waveform for selection"""
        if event.inaxes == self.wave_ax and self.wave_selection_start is not None:
            self.wave_selection_end = event.xdata
            # Ensure start < end
            if self.wave_selection_start > self.wave_selection_end:
                self.wave_selection_start, self.wave_selection_end = self.wave_selection_end, self.wave_selection_start
            
            # Update visualization
            self._update_waveform_plot(self.current_audio_data)
    
    def save_visualization_data(self, viz_type):
        """Save the selected part of visualization as a WAV file"""
        try:
            if viz_type == 'spectrogram':
                if self.spec_selection_start is None or self.spec_selection_end is None:
                    self.log("No spectrogram selection to save")
                    return
                
                # Convert time selection to samples
                start_sample = int(self.spec_selection_start * self.sample_rate)
                end_sample = int(self.spec_selection_end * self.sample_rate)
                
            elif viz_type == 'waveform':
                if self.wave_selection_start is None or self.wave_selection_end is None:
                    self.log("No waveform selection to save")
                    return
                
                # Use sample indices directly
                start_sample = max(0, int(self.wave_selection_start))
                end_sample = min(len(self.current_audio_data) - 1, int(self.wave_selection_end))
            
            # Check bounds
            if start_sample < 0:
                start_sample = 0
            if end_sample >= len(self.current_audio_data):
                end_sample = len(self.current_audio_data) - 1
            
            # Extract the selected data
            selected_data = self.current_audio_data[start_sample:end_sample+1]
            
            if len(selected_data) == 0:
                self.log("Selected region is too small")
                return
            
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )
            
            if not filename:
                return
            
            # Normalize data if needed
            max_val = np.max(np.abs(selected_data))
            if max_val > 1.0:
                selected_data = selected_data / max_val
            
            # Save as WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                
                # Convert float to PCM
                data_int = (selected_data * 32767).astype(np.int16)
                wf.writeframes(data_int.tobytes())
            
            self.log(f"{viz_type.capitalize()} selection saved to {filename}")
            
        except Exception as e:
            self.log(f"Error saving selection: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.server_button['text'] == "Stop Server":
            self.stop_server()
        self.root.destroy()
    
    def run(self):
        """Run the main application loop"""
        self.root.mainloop()

if __name__ == "__main__":
    server = AudioServer()
    server.run()
