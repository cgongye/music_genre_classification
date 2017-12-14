import argparse
import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from feature_extraction import GENRE_LIST
from feature_extraction import generate_features

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="This is a simple demo that takes an audio file and print its predicted genre")
    argparser.add_argument("-f", default="demo.wav", help="wav file to test")
    args = argparser.parse_args()
    os.system("ffmpeg -ss 60 -t 30 -i %s -acodec pcm_s16le -ac 1 -ar 16000 out.wav" %args.f)
    # load model
    clf = joblib.load("model.out")

    # generate feature
    generate_features("out.wav")

    # scale the feature
    X = np.load("X.npy")
    ff = 'out.features.npy'
    feature = np.load(ff).reshape(1, 122)
    scaler = StandardScaler()
    scaler.fit(X)
    feature = scaler.transform(feature)

    # predict
    label = clf.predict(feature)[0]
    print "The prediction is:", GENRE_LIST[label]
    os.system('rm out.*')
