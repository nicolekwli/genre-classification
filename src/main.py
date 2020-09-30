import sys
import os
import random
import math
import numpy as np
import scipy.IO.wavfile as wav

from file import AudioFile

# Gets distance and finds neighbors
def get_neighbours(train, instance, k):
    distances = []
    for x in range (len(train)):
        dist = distance(train[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((train[x][2], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []

    for x in range(k):
        neighbors.append(distances[x][0])

    return neighbors


def nearest_class(neighbors):
    votes = {}

    for x in range (len(neighbors)):
        response = neighbors[x]
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1

    sorter = sorted(votes.items(), key= operator.itemgetter(1), reverse=True)
    return sorter[0][0]

# accuracy for knn
def accuracy(test, predictions):
    correct = 0
    for x in range (len(test)):
        if test[x][-1] == predictions[x]:
            correct += 1
    return 1.0 * correct / len(test)

# attempt to classify with knn
if __name__ == '__main__':
    # path = "../assets/test.mp3"
    # audio = AudioFile(path)
    # audio.waveform_zoom()
    # print(audio.zeroCross)
    # audio.mfcc()
    # audio.chroma_freq()

    # Extract features and dump into .dat
    directory = ""




