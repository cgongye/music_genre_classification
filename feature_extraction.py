import glob
import os
import time

import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc

GENRE_DIR = "/home/cheng/Desktop/genres"
GENRE_LIST = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


def write_ceps(features, f):
    """
    This function write the features of wav file f as f.featrues.npy

    :param features: an one dimension nparray containing the extracted features
    :param f: file name
    :return: None
    """
    base_f, ext = os.path.splitext(f)
    data_f = base_f + ".features"
    np.save(data_f, features)


def generate_features(f):
    """
    This function generate the features of the wav file f

    :param f: a wav file
    :return: None
    """
    # read wav file
    sample_rate, X = scipy.io.wavfile.read(f)

    # so that no dividing by zero
    X[X == 0] = 1

    # mfcc transform
    mfccs, logs, ffts = mfcc(X, nfft=551)

    # trim the beginning and the end part of the wav file
    num_features = len(mfccs)
    start = int(num_features / 9)
    end = int(num_features * 8 / 9)

    # generate mean and std of the mfcc coefficient along the columns
    features_mean = np.mean(mfccs[start:end], axis=0)
    features_std = np.std(mfccs[start:end], axis=0)
    features_result = np.append(features_mean, features_std)

    # generate mean of the log energy on mel dimension  along the columns
    log_result = np.mean(logs, axis=0)

    # trim the beginning and the end part of the wav file
    mfccs, logs, ffts = mfcc(X, nfft=8)
    num_features = len(ffts)
    start = int(num_features / 9)
    end = int(num_features * 8 / 9)

    # generate mean and std of the fft along the columns
    specs_mean = np.mean(ffts[start:end], axis=0)
    specs_std = np.std(ffts[start:end], axis=0)
    specs_result = np.append(specs_mean, specs_std)

    # write the result to a file
    result = np.append(features_result, specs_result)
    write_ceps(np.append(result, log_result), f)


def read_features():
    """
    This function load the saved features in an nparrays. If X.npy and y.npy exist, it will load them
    else it will load it from the GENRE_DIR

    :return: X the nparray that contains the features, one sample each row. y, the one dimension nparray
            that contains the labels
    """
    if os.path.isfile("X.npy") and os.path.isfile("y.npy"):
        return np.load("X.npy"), np.load("y.npy")
    X = []
    y = []
    for label, genre in enumerate(GENRE_LIST):
        for fn in glob.glob(os.path.join(GENRE_DIR, genre, "*.features.npy")):
            features = np.load(fn)
            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    np.save("X", X)
    np.save("y", y)
    return X, y


if __name__ == "__main__":

    start = time.time()

    print "Starting ceps generation..."
    for subdir, dirs, files in os.walk(GENRE_DIR):
        for file in files:
            path = subdir + '/' + file
            if path.endswith("wav"):
                generate_features(path)

    print "Time used: ", time.time() - start
