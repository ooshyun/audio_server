import sounddevice as sd
# import soundfile as sf
import numpy as np
import time

def list_audio_devices() -> dict:
    """List all available audio input and output devices"""
    print("Available Audio Devices:")
    print("-----------------------")
    
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append("Input")
        if device['max_output_channels'] > 0:
            device_type.append("Output")
            
        print(f"Device {i}: {device['name']}")
        print(f"  Type: {' & '.join(device_type)}")
        print(f"  Input Channels: {device['max_input_channels']}")
        print(f"  Output Channels: {device['max_output_channels']}")
        print(f"  Default Sample Rate: {device['default_samplerate']}")
        print()
    
    default_input = sd.query_devices(kind='input')
    default_output = sd.query_devices(kind='output')
    
    print(f"Default Input Device: {default_input['name']}")
    print(f"Default Output Device: {default_output['name']}")
    
    device_dict = {}
    for i, device in enumerate(devices):
        device_dict[i] = device
    return device_dict

def get_intput_stream(device_index: int, 
                      channels: int, 
                      sample_rate: int, 
                      blocksize: int) -> sd.InputStream:
    # Stream for real-time processing
    return sd.InputStream(
        device=device_index,
        channels=channels,
        samplerate=sample_rate)

def get_output_stream(device_index: int,
                        channels: int,
                        sample_rate: int,
                        blocksize: int) -> sd.OutputStream:
        return sd.OutputStream(
            device=device_index,
            channels=channels,
            samplerate=sample_rate)

# def record_audio(device_index=None, duration=5, save_to_file=False, filename="recording.wav"):
#     """
#     Record audio from specified microphone device
    
#     Parameters:
#     device_index (int): Device index to use (None for default)
#     duration (float): Recording duration in seconds
#     save_to_file (bool): Whether to save the recording to a file
#     filename (str): Name of file to save recording to
    
#     Returns:
#     numpy.ndarray: Audio data
#     """
#     RATE = 44100
#     CHANNELS = 1
    
#     # If no device specified, use default
#     if device_index is None:
#         device_info = sd.query_devices(kind='input')
#         device_index = sd.default.device[0]
#     else:
#         device_info = sd.query_devices(device_index)
    
#     # Check if it's an input device
#     if device_info['max_input_channels'] <= 0:
#         raise ValueError(f"Device {device_index} has no input channels!")
    
#     print(f"Recording from: {device_info['name']}")
#     print(f"Sample rate: {RATE} Hz, Duration: {duration} seconds")
    
#     # Record audio
#     print("Recording started...")
#     audio_data = sd.rec(
#         int(duration * RATE),
#         samplerate=RATE,
#         channels=CHANNELS,
#         device=device_index,
#         dtype='float32'
#     )
    
#     # Wait for recording to complete
#     sd.wait()
#     print("Recording finished!")
    
#     # Save to file if requested
#     if save_to_file:
#         sf.write(filename, audio_data, RATE)
#         print(f"Recording saved to {filename}")
    
#     return audio_data

def real_time_audio_processing(device_index=None, duration=10, callback=None):
    """
    Capture and process audio in real-time
    
    Parameters:
    device_index (int): Device index to use (None for default)
    duration (float): Duration to run in seconds
    callback (function): Function to call for each chunk of audio data
                         Should accept a numpy array of audio samples
    """
    RATE = 44100
    CHANNELS = 1
    BLOCKSIZE = 1024
    
    # If no device specified, use default
    if device_index is None:
        device_index = sd.default.device[0]
    
    # Default callback if none provided
    if callback is None:
        def callback(indata, frames, time, status):
            if status:
                print(status)
            
            # Calculate volume (RMS)
            audio_chunk = indata[:, 0]  # Get first channel
            rms = np.sqrt(np.mean(np.square(audio_chunk)))
            print(f"Current volume: {rms:.2f}")
    
    # Stream for real-time processing
    stream = sd.InputStream(
        device=device_index,
        channels=CHANNELS,
        samplerate=RATE,
        blocksize=BLOCKSIZE,
        callback=callback
    )
    
    print(f"Processing audio from device {device_index} in real-time...")
    
    # Start the stream
    with stream:
        try:
            # Keep the stream active for the specified duration
            time.sleep(duration)
        except KeyboardInterrupt:
            print("\nStopped by user")
    
    print("Audio processing completed")

# Custom callback function for the real-time processing
def spectral_analysis_callback(indata, frames, time, status):
    """Example callback to perform FFT on audio chunk"""
    if status:
        print(status)
    
    # Get audio data (first channel)
    audio_chunk = indata[:, 0]
    
    # Apply window function to reduce spectral leakage
    windowed = audio_chunk * np.hanning(len(audio_chunk))
    
    # Perform FFT
    spectrum = np.abs(np.fft.rfft(windowed))
    
    # Convert to dB
    spectrum_db = 20 * np.log10(spectrum + 1e-10)  # Add small value to avoid log(0)
    
    # Get the frequency bins
    freqs = np.fft.rfftfreq(len(audio_chunk), 1.0/44100)
    
    # Print maximum frequency component (simple example)
    max_idx = np.argmax(spectrum)
    max_freq = freqs[max_idx]
    print(f"Dominant frequency: {max_freq:.1f} Hz, Magnitude: {spectrum_db[max_idx]:.1f} dB")

def visualize_spectrum_callback(indata, frames, time, status):
    """
    Alternative callback to visualize the audio spectrum in terminal
    This is a simple visualization, you'd use matplotlib for better visuals
    """
    if status:
        print(status)
    
    # Get audio data (first channel)
    audio_chunk = indata[:, 0]
    
    # Apply window and perform FFT
    windowed = audio_chunk * np.hanning(len(audio_chunk))
    spectrum = np.abs(np.fft.rfft(windowed))
    
    # Convert to dB and normalize
    spectrum_db = 20 * np.log10(spectrum + 1e-10)
    normalized = (spectrum_db + 80) / 80  # Normalize to 0-1 range (assuming min is -80dB)
    normalized = np.clip(normalized, 0, 1)
    
    # Simple terminal visualization with ASCII
    width = 60  # Width of the visualization
    bins = 20   # Number of frequency bins to show
    
    # Downsample spectrum to the desired number of bins
    bin_size = len(normalized) // bins
    downsampled = [np.max(normalized[i:i+bin_size]) for i in range(0, len(normalized), bin_size)][:bins]
    
    # Clear terminal
    print("\033[H\033[J", end="")
    
    # Print the spectrum
    print("Audio Spectrum:")
    for amp in downsampled:
        bar_length = int(amp * width)
        print(f"{'█' * bar_length}{' ' * (width - bar_length)} | {amp:.2f}")

# Example of how to use the functions
if __name__ == "__main__":
    # List all audio devices
    devices = list_audio_devices()
    print(devices)
    
    # Choose a device (use default if unsure)
    mic_index = None  # Replace with a specific device index if needed
    
    # # Option 1: Record audio for a set duration
    # audio_data = record_audio(device_index=mic_index, duration=3, save_to_file=True)
    
    # # Option 2: Process audio in real-time with spectral analysis
    # real_time_audio_processing(
    #     device_index=mic_index, 
    #     duration=10, 
    #     callback=spectral_analysis_callback
    # )
    
    # # Option 3: Real-time spectrum visualization
    # real_time_audio_processing(
    #     device_index=mic_index,
    #     duration=10,
    #     callback=visualize_spectrum_callback
    # )