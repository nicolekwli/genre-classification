import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import sklearn

# helpers ---------------------------
def normalise(data, axis=0):
    return sklearn.preprocessing.minmax_scale(data, axis=axis)

class AudioFile():
    def __init__(self, path):
        self.path = path
        self.sr = 44100
        self.data, self.sr = self.load(self.path, self.sr) # audio time series as numpy array
        self.zeroCross = self.zero_crossings()
        self.mfccs = self.mfcc()

    def load(self, path, rate):
        return librosa.load(path, rate)

    def play(self):
        ipd.Audio(self.path)

    # plots --------------------------------------
    def waveform(self):
        plt.figure()
        plt.subplot(3, 1, 1)
        librosa.display.waveplot(self.data, sr=self.sr)
        plt.show()

    # TODO: add range as parameter
    def waveform_zoom(self):
        plt.figure()
        plt.subplot(3, 1, 1)
        librosa.display.waveplot(self.data[15500:16000], sr=self.sr)
        plt.show()

    def spectrogram(self):
        X = librosa.stft(self.data)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure()
        plt.subplot(3, 1, 1)
        librosa.display.specshow(Xdb, sr=self.sr, x_axis='time', y_axis='hz') # or y='log'
        plt.colorbar()
        plt.show()

    # features -------------------------------------
    # TODO: add range as parameter
    def zero_crossings(self):
        return sum(librosa.zero_crossings(self.data[15500:16000], pad=False))

    # center of mmass for a sound (CURRENTLY A PLOT)
    def spec_cent(self):
        plt.figure()
        plt.subplot(3, 1, 1)

        # weighted mean of the frequencies present in the sound
        spectral_centroids = librosa.feature.spectral_centroid(self.data, sr=self.sr)[0]

        # to check shape: spectral_centroids.shape

        # Computing the time variable for visualization
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)

        # normamlise 
        normal = normalise(spectral_centroids, 0)

        #Plotting the Spectral Centroid along the waveform
        librosa.display.waveplot(self.data, sr=self.sr, alpha=0.4)
        plt.plot(t, normal, color='r')
        plt.show()

    # measure shape of signal
        # represents the frequency below which a specified percentage of the total spectral energy
        # CURRENTL A PLOT
    def spec_rolloff(self):
        plt.figure()
        plt.subplot(3, 1, 1)

        # weighted mean of the frequencies present in the sound
        spectral_centroids = librosa.feature.spectral_centroid(self.data, sr=self.sr)[0]
        # Computing the time variable for visualization
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)

        # rolloff freq for each frame
        spectral_rolloff = librosa.feature.spectral_rolloff(self.data+0.01, sr=self.sr)[0]

        librosa.display.waveplot(self.data, sr=self.sr, alpha=0.4)

        plt.plot(t, normalise(spectral_rolloff), color='r')
        plt.show()

    # Mel frequency cepstral coefficients
        # set of features that describe the overall spahe of the spectral envelope
    def mfcc(self):
        mfccs = librosa.feature.mfcc(self.data, sr=self.sr) # mfccs.shape (20, 97) - 20 MFCC s over 97 frames

        plt.figure()
        plt.subplot(3, 1, 1)
        librosa.display.specshow(mfccs, sr=self.sr, x_axis='time')
        plt.show()

        return mfccs

        
    def mfcc_scale():
        pass


    

