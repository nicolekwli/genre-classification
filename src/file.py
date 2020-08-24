import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

class audioFile:
    def __init__(self, path):
        self.path = path
        self.data = load(path) # audio time series as numpy array

    def load():
        return librosa.load(self.path, sr=44100)

    def play():
        ipd.Audio(self.path)

    def waveform():
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(self.data, sr=44100)

    def spectrogram():
        X = librosa.stft(self.data)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=44100, x_axis='time', y_axis='hz') # or y='log'
        plt.colorbar()
