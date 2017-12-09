from sklearn.model_selection import cross_val_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from ceps import read_ceps
from utils import GENRE_LIST
from joblib import Parallel, delayed
import multiprocessing

genre_list = GENRE_LIST
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def print_scores(scores, p, name):
    print("Accuracy: %0.3f (+/- %0.3f) %s %d" % (scores.mean(), scores.std() * 2, name, p))


def model_score(clf, X, y, p, name):
    scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    print_scores(scores, p, name)


def process(i, tmp_X, y, name):
    c = 2 ** i
    model_score(svm.SVC(kernel='linear', C=c), tmp_X, y, c, name + ' ' + 'linear')
    model_score(svm.SVC(kernel='rbf', C=c), tmp_X, y, c, name + ' ' + 'rbf')
    model_score(svm.SVC(kernel='poly', C=c), tmp_X, y, c, name + ' ' + 'poly')


if __name__ == "__main__":
    X, y = read_ceps(genre_list)
    poly = PolynomialFeatures(3)
    mfcc = poly.fit_transform(X[:, :26])
    X = np.append(mfcc, poly.fit_transform(X[:, 26:]), axis=1)
    # num_cores = multiprocessing.cpu_count()

    # Parallel(n_jobs=num_cores)(delayed(process)(i,X, y,'c') for i in range(16))

    # i = 3
    # print "Logistic regression degree", i
    # poly = PolynomialFeatures(i)
    model_score(LogisticRegression(), X, y, 123456, 'seed')
    # for i in [x for x in range(1000)]:
    #     model_score(RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=i),X,y,i,'n')
    # model_score(MLPClassifier(),X,y,1,'a')
