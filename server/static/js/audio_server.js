console.log(typeof FFT);  // Should be "function"

// audio parameters
let _windowSize = 1;
let _hopSize = 0.5;
let _fft_size = 256;
let _overlap = 128;
let _num_mels = 80;
let _f_min = 0;
let _f_max = 8000;

// parameters by using audio parameters
let _num_fft_bins = 129;

// let _sampleRate = 16000;
// let _num_frames = 126;

let _sampleRate = 44100;
let _num_frames = 345;

function getSampleRate() {
    return _sampleRate;
}

function getNumFFT() {
    return _fft_size;
}

function getNumFFTbins() {
    return _num_fft_bins;
}

function getNumFrames() {
    return _num_frames;
}

// Update Parameters
function updateAudioConfig() {
    fetch('/get_audio_config')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();  
        })
        .then(data => {
            console.log(data)
            if (data && data.audio) {
                _sampleRate = data.audio.sample_rate;
                _windowSize = data.audio.window_size;
                _hopSize = data.audio.hop_size;
                _fft_size = data.audio.fft_size;
                _overlap = data.audio.overlap;
                _num_mels = data.audio.num_mels;
                _f_min = data.audio.f_min;
                _f_max = data.audio.f_max;

                _num_fft_bins = _fft_size / 2 + 1;
                _num_frames = Math.floor((_windowSize * _sampleRate / _overlap) + 1);
                console.log("Audio Config Updated: ")
                console.log("  - Sample Rate: ", _sampleRate)
                console.log("  - Window Size: ", _windowSize)
                console.log("  - Hop Size: ", _hopSize)
                console.log("  - FFT Size: ", _fft_size)
                console.log("  - Overlap: ", _overlap)
                console.log("  - Num Mels: ", _num_mels)
                console.log("  - F Min: ", _f_min)
                console.log("  - F Max: ", _f_max)
                console.log("Derived Config: ");
                console.log("  - Num frame: ",  _num_frames);
                console.log("  - Num FFT: ", _num_fft_bins);
            } else {
                console.error('Audio data not found in the response');
                console.log('Data structure:', JSON.stringify(data, null, 2));
            }
        })
        .catch(error => console.error('Failed to fetch data:', error));
}

// Waveform update function
function updateWaveform(audioData) {
    // console.log('Received audio data:', audioData.length); // Why send 64 buffer, but it is 1024?
    if (!audioData || !audioData.length) return;
    waveformChart.data.datasets[0].data = audioData;
    waveformChart.update();
}

// Simple spectrogram update function
function updateFFT(FFTData) {
    if (!FFTData || !FFTData.length) return;
    // console.log('Received FFT data:', FFTData.length);
    // This is a simplified spectrogram. In reality, you'd need to perform FFT.
    fftChart.data.datasets[0].data = FFTData
    fftChart.update();
}

// Improved STFT implementation
function performSTFT(audioData, windowSize, hopSize, numFrames) {
    if (!audioData || audioData.length < windowSize) return [];
    // TODO(shoh): fft is not working, real size is 1
    const fft = new FFT(windowSize, _sampleRate); // Assuming 16kHz sample rate
    // console.log("FFT Size: ", fft.real); // length is 1

    // Use only the most recent audio data needed
    const requiredSamples = (numFrames - 1) * hopSize + windowSize;
    const recentAudio = audioData.length > requiredSamples 
        ? audioData.slice(audioData.length - requiredSamples) 
        : audioData;
    
    // Create result array for STFT data
    const stftResult = [];
    
    // Process each frame
    const actualFrames = Math.min(
        numFrames, 
        Math.floor((recentAudio.length - windowSize) / hopSize) + 1
    );
    
    // console.log("Actual Frames: ", actualFrames);

    for (let frame = 0; frame < actualFrames; frame++) {
        // Extract frame data
        const startIdx = frame * hopSize;
        if (startIdx + windowSize > recentAudio.length) break;
        
        const frameData = recentAudio.slice(startIdx, startIdx + windowSize);
        
        // Apply window function (Hanning window)
        const windowedFrame = applyWindow(frameData, windowSize);
        
        // Perform FFT
        fft.forward(windowedFrame);
        
        // Get magnitudes (only need first half due to symmetry)
        const magnitudes = new Float32Array(windowSize / 2);
        for (let i = 0; i < windowSize / 2; i++) {
            magnitudes[i] = Math.sqrt(fft.real[i] * fft.real[i] + fft.imag[i] * fft.imag[i]);
        }
        
        // Apply log scale for better visualization
        const logMagnitudes = Array.from(magnitudes).map(m => 
            Math.max(0, 20 * Math.log10(m + 1e-6))
        );
        
        stftResult.push(logMagnitudes);
    }
    
    return stftResult;
}

// Apply Hanning window function
function applyWindow(frame, windowSize) {
    const windowed = new Float32Array(windowSize);
    for (let i = 0; i < frame.length; i++) {
        // Hanning window: 0.5 * (1 - cos(2π*n/(N-1)))
        const windowValue = 0.5 * (1 - Math.cos(2 * Math.PI * i / (windowSize - 1)));
        windowed[i] = frame[i] * windowValue;
    }
    return windowed;
}

// Update STFT visualization
function updateSpectrogramTmp(audioData) {
    // print length of audioData
    // console.log(audioData.length); // _windowSize*_sampleRate = 16000

    if (!audioData || audioData.length < _windowSize*_sampleRate)  {
        console.log("No valid audio data received:", audioData);
        return;
    }
    
    // Perform STFT
    const stftData = performSTFT(audioData,
                                 _windowSize,
                                 _hopSize,
                                 _num_frames);
    // console.log(stftData.length); // 31489
    // Skip if we don't have enough data
    if (!stftData.length) return;
    
    // Find min/max for better normalization
    let min = Infinity;
    let max = -Infinity;
    
    stftData.forEach(frame => {
        frame.forEach(value => {
            min = Math.min(min, value);
            max = Math.max(max, value);
        });
    });
    
    // Avoid division by zero and ensure some dynamic range
    min = min === Infinity ? 0 : min;
    max = max === -Infinity ? 1 : max;
    const range = Math.max(1, max - min);
    
    // Convert to matrix chart format - COMPLETELY REPLACE existing data
    const heatmapData = [];
    // console.log(_fft_size, stftData[0].length);
    const numBins = Math.min(_fft_size, stftData[0].length); // Limit frequency bins
    
    // console.log("Length of STFT: ", stftData.length);
    // console.log("Number of Bins: ", numBins);

    for (let frameIdx = 0; frameIdx < stftData.length; frameIdx++) {
        for (let binIdx = 0; binIdx < numBins; binIdx++) {
            // Get normalized value (0-255)
            const normalizedValue = Math.round(((stftData[frameIdx][binIdx] - min) / range) * 255);
            
            // Add data point with proper orientation
            heatmapData.push({
                x: frameIdx,                 // Time on x-axis
                y: numBins - binIdx - 1,     // Frequency on y-axis (inverted)
                v: normalizedValue           // Color intensity
            });
        }
    }
    
    // console.log(heatmapData.length); // 896 = 128x7
    stftChart.data.datasets[0].data = heatmapData;
    stftChart.update();
}

function updateSpectrogram(stftData)  {
    if (!stftData || !stftData.length) return;
    // console.log('Received stftData data:', stftData.length);
    const heatmapData = [];
    numBins = Math.min(_num_fft_bins, stftData[0].length); // Limit frequency bins
    for (let frameIdx = 0; frameIdx < stftData.length; frameIdx++) {
        for (let binIdx = 0; binIdx < numBins; binIdx++) {
                // Get normalized value (0-255)            
                // Add data point with proper orientation
                heatmapData.push({
                    x: frameIdx,                        // Time on x-axis
                    y: numBins - binIdx - 1,            // Frequency on y-axis (inverted)
                    v: stftData[frameIdx][binIdx]       // Color intensity
                });
        }
    }
    // console.log('Draw stftData: ', heatmapData.length);
    // console.log(stftChart.data.datasets[0].data.length);
    // console.log(heatmapData);
    stftChart.data.datasets[0].data = heatmapData;
    stftChart.update();
}
