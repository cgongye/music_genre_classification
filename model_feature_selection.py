import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from feature_extraction import read_features

####################################################################################
#  changing parameters below is not recommended unless you know what you are doing #
####################################################################################
# for MLP
MAX_NUS = 110

# for cross validation
CV = 5
NUM_SEEDS = 50

# for RF
MAX_TREES = 200

# for LG
MAX_C = 100


####################################################################################

class ModelSelector():
    def __init__(self, X, y, clf):
        self.X = X
        self.y = y
        self.clf = clf

    @staticmethod
    def print_scores(score_mean, score_std, p, name):
        print("Accuracy: %0.3f (+/- %0.3f). The %s is %d" % (score_mean, score_std, name, p))

    def model_score(self, p, name):
        scores = cross_val_score(self.clf, self.X, self.y, cv=CV, n_jobs=-1)
        score_mean, score_std = scores.mean(), scores.std()
        self.print_scores(score_mean, score_std, p, name)
        return scores.mean()


def parse_features(type):
    X, y = read_features()
    mfcc = X[:, :26]
    fft = X[:, 26:42]
    log = X[:, 42:]
    if type == 1:
        X = mfcc
    elif type == 2:
        X = fft
    elif type == 3:
        X = log
    elif type == 4:
        X = np.append(mfcc, fft, axis=1)
    elif type == 5:
        X = np.append(mfcc, log, axis=1)
    elif type == 6:
        X = np.append(fft, log, axis=1)
    return X, y


def test_rf(X, y):
    model = ModelSelector(X, y, RandomForestClassifier())
    scores = []
    indexes = []
    for i in range(MAX_TREES / 10):
        sub_scores = []
        num_trees = i * 10 + 10
        print "Number of trees: ", num_trees
        for j in range(NUM_SEEDS):
            model.clf = RandomForestClassifier(n_estimators=num_trees, random_state=j,n_jobs=-1)
            sub_scores.append(model.model_score(j, "SEED"))
        scores.append(sub_scores)
        indexes.append(num_trees)
    scores = list(np.max(np.array(scores), axis=1))
    plot_scores(scores, indexes, "Accuracy", "Number of trees",
                "Accuracy", "Accuracies of different number of estimators of RF", "RF.png")


def plot_scores(scores, indexes, label, xlabel, ylabel, title, filename):
    plt.plot(indexes, scores, label=label, lw=1.5, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig("./graph/" + filename)
    plt.show()
    # plt.clf()


def test_mlp(X, y):
    # MLP needs normalized features
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    model = ModelSelector(X, y, MLPClassifier())
    scores = []
    indexes = []
    for i in range(MAX_NUS / 10):
        sub_scores = []
        num_neurons = i * 10 + 10
        print "Number of neurons: ", num_neurons
        for j in range(NUM_SEEDS):
            model.clf = MLPClassifier(hidden_layer_sizes=(num_neurons, num_neurons), random_state=j)
            sub_scores.append(model.model_score(j, "SEED"))
        scores.append(sub_scores)
        indexes.append(num_neurons)
    scores = list(np.max(np.array(scores), axis=1))
    plot_scores(scores, indexes, "Accuracy", "Number of neurons",
                "Accuracy", "Accuracies of different number of neurons of MLP", "MLP.png")


def test_rg(X, y, lift):
    # lifting: we are only lifting mfcc features according to test result
    poly = PolynomialFeatures(lift)
    mfcc = poly.fit_transform(X[:, :26])
    X = np.append(mfcc, X[:, 26:], axis=1)

    # sag needs normalized features
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    print "Caclculation scores..."
    model = ModelSelector(X, y, LogisticRegression(solver="sag", n_jobs=-1))
    model.model_score(1,'C')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train a model do a validation analysis of the model')
    parser.add_argument('--features', default=7, type=int, help=
    """
    1: mfcc
    2: fft
    3: log-mel
    4: mfcc + fft
    5: mfcc + log-mel
    6: fft + log-mel
    7: mfcc + fft + log-mel
    """)
    parser.add_argument('--N', default=-1, type=int, help="number of cores")
    parser.add_argument('--classifier', default=2, type=int, help=
    """
    1: Random Forest
    2: Logistic Regression (very slow)
    3: Multi-layer Perceptron
    """)
    parser.add_argument('--lift', default=3, type=int,
                        help="degree of polynomial features, only works with logistic regression")
    args = parser.parse_args()

    # read and select features
    X, y = parse_features(args.features)

    # test the classifier
    if args.classifier == 1:
        test_rf(X, y)
    elif args.classifier == 2:
        if args.features in [2, 3, 6]:
            warnings.warn("Lifting will not work because mfcc features are not used")
            test_rg(X, y, 1)
        else:
            test_rg(X, y, args.lift)
    elif args.classifier == 3:
        test_mlp(X, y)
