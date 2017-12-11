import argparse
import itertools

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from feature_extraction import GENRE_LIST, read_features


class category_result:
    """
    class for calculating and storing result of each category

    """
    mean_fpr = np.linspace(0, 1, 100)

    def __init__(self):
        self.fprs = []
        self.tprs = []
        self.aucs = []

    def append(self, y_test, proba, label):
        fpr, tpr, thresholds = roc_curve(y_test, proba, label)
        self.fprs.append(fpr)
        self.tprs.append(tpr)
        self.aucs.append(auc(fpr, tpr))

    def print_result(self, label):
        for fold in range(len(self.fprs)):
            plt.plot(self.fprs[fold], self.tprs[fold], lw=1.5, alpha=0.3,
                     label='ROC fold %d AUC = %0.2f' % (fold, self.aucs[fold]))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
                 label='Random', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROCs of %s, Mean AUC = %0.2f $\pm$ %0.2f'
                  % (GENRE_LIST[label], np.mean(self.aucs), np.std(self.aucs)))
        plt.legend(loc="lower right")
        plt.savefig('./graph/validation/%s.png' % GENRE_LIST[label])
        # plt.show()
        plt.clf()


class fold_result:
    """
    class for calculating and storing result of each fold

    """

    def __init__(self):
        self.train_scores = []
        self.test_scores = []
        self.confussion_m = np.zeros((10, 10))

    def append(self, clf, X_train, y_train, X_test, y_test):
        self.train_scores.append(clf.score(X_train, y_train))
        self.test_scores.append(clf.score(X_test, y_test))
        self.confussion_m += confusion_matrix(y_test, clf.predict(X_test))

    def print_score(self):
        print("Train set accuracy: %0.3f (+/- %0.3f)"
              % (np.mean(self.train_scores), np.std(self.train_scores)))
        print("Test set accuracy: %0.3f (+/- %0.3f)"
              % (np.mean(self.test_scores), np.std(self.test_scores)))
        # Normalize conffusion matrix
        self.confussion_m = self.confussion_m.astype('float') / self.confussion_m.sum(axis=1)[:, np.newaxis]
        plt.imshow(self.confussion_m, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Conffusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(GENRE_LIST))
        plt.xticks(tick_marks, GENRE_LIST, rotation=45)
        plt.yticks(tick_marks, GENRE_LIST)

        fmt = '.2f'
        thresh = self.confussion_m.max() / 2.
        for i, j in itertools.product(range(self.confussion_m.shape[0]), range(self.confussion_m.shape[1])):
            plt.text(j, i, format(self.confussion_m[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if self.confussion_m[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("./graph/validation/cm.png")
        plt.show()


def train_model(X, y, out):
    """
    This function train the model and save the pickled result

    :param X: nparray, one row is one sample
    :param y: nparray, labels
    :param out: output filename
    :param N: number of CPU cores to use
    :param digit: digit of the captcha
    :param classifier: which classifier to use
    :return: None
    """
    clf = MLPClassifier(hidden_layer_sizes=(110, 110), random_state=17)
    clf.fit(X, y)
    # output name
    with open(out, 'w') as f:
        joblib.dump(clf, f)
    print "Model saved in", out


def validate_model(X, y):
    """
    This function validate the model by K-fold cross validation and print the ROC curves
    in the graph/validation

    :param X: nparray, one row is one sample
    :param y: nparray, labels
    :param out: output filename
    :param N: number of CPU cores to use
    :param digit: digit of the captcha
    :param classifier: which classifier to use
    :return: None
    """
    # K-fold cross validation
    folds = KFold(n_splits=5, shuffle=True, random_state=108).split(X)
    fold_r = fold_result()
    labels = np.unique(y)
    category_rs = [None] * len(labels)
    for label in labels:
        category_rs[label] = category_result()
    for train, test in folds:
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        clf = MLPClassifier(hidden_layer_sizes=(110, 110), random_state=17)
        clf.fit(X_train, y_train)
        probas = clf.predict_proba(X_test)
        fold_r.append(clf, X_train, y_train, X_test, y_test)
        for label in labels:
            category_rs[label].append(y_test, probas[:, label], label)
    fold_r.print_score()
    # print and save ROCS
    for label in labels:
        category_rs[label].print_result(label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train a model using MLP and do a validation analysis of the model')
    parser.add_argument('--validate', default=1, type=int, help="validate the model or not")
    parser.add_argument('--out', default="model.out", help='trained model')

    args = parser.parse_args()

    # load processed data
    X, y = read_features()

    # MLP needs normalized features
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # train the model
    train_model(X, y, args.out)

    # validate the model
    if (args.validate):
        validate_model(X, y)
