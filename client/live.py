import numpy as np
import argparse
import sys

import sounddevice as sd

def parse_audio_device(device):
    if device is None:
        return device
    try:
        return int(device)
    except ValueError:
        return device


def query_devices(device, kind):
    try:
        caps = sd.query_devices(device, kind=kind)
    except ValueError:
        message = f"Invalid {kind} audio interface {device}.\n"
        message += (
            "If you are on Mac OS X, try installing Soundflower "
            "(https://github.com/mattingalls/Soundflower).\n"
            "You can list available interfaces with `python3 -m sounddevice` on Linux and OS X, "
            "and `python.exe -m sounddevice` on Windows. You must have at least one loopback "
            "audio interface to use this.")
        print(message, file=sys.stderr)
        sys.exit(1)
    return caps

def main():
    print(sd.query_devices())
    sample_rate = 16000

    device_in = parse_audio_device(1)
    channels_in = 1
    stream_in = sd.InputStream(
        device=device_in,
        samplerate=sample_rate,
        channels=channels_in)

    device_out = parse_audio_device(6)
    channels_out = 1
    stream_out = sd.OutputStream(
        device=device_out,
        samplerate=sample_rate,
        channels=channels_out)

    stream_in.start()
    stream_out.start()
    while True:
        try:
            # length = streamer.total_length if first else streamer.stride
            length = 64
            frame, overflow = stream_in.read(length)
            print(type(frame), frame.dtype)
            np.clip(frame, -1, 1, out=frame)
            underflow = stream_out.write(frame)
            if overflow or underflow:
                ...
        except KeyboardInterrupt:
            print("Stopping")
            break
    stream_out.stop()
    stream_in.stop()


if __name__ == "__main__":
    main()
