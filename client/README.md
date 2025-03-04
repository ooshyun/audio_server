# Client for audio server
First of all, you need to install portaudio to use sounddevice python library

```shell
sudo apt install portaudio2
```

After that, install requirements for python packages and run client python code.

```shell
pip3 install -r requirements.txt
python client.py
```

If you want to test streaming, then use live code

```shell
python live.py
```