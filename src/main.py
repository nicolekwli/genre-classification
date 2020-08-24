import sys
import numpy as np

from file import AudioFile



if __name__ == '__main__':
    path = "../assets/test.mp3"
    audio = AudioFile(path)
    audio.spec_rolloff()
    print(audio.zeroCross)

