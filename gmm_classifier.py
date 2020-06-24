from mnist_dataset import get_mnist
import pdb
import numpy as np
import math


def log_gaussian(o, mu, r):
    compute = (-0.5 * np.log(r) - np.divide(np.square(o - mu), 2 * r) - 0.5 * np.log(2 * math.pi)).sum()
    return compute


class SingleGauss:
    def __init__(self):
        self.dim = None
        self.mu = None
        self.r = None

    def train(self, x):
        data = np.vstack(x)
        self.mu = np.mean(x, axis=0)
        self.r = np.mean(np.square(np.subtract(x, self.mu)), axis=0)

    def loglike(self, x):
        return log_gaussian(x, self.mu, self.r)


class gaussian_classifier:
    def __init__(self):
        self.n_classes = None
        self.mu = None
        self.r = None

    def train(self, x, y):
        self.n_classes = len(set(y))
        self.classifiers = [SingleGauss() for _ in range(self.n_classes)]
        for i in range(self.n_classes):
            mask = np.where(y == i)[0]
            xtmp = x[mask]
            ytmp = y[mask]
            self.classifiers[i].train(x_train)

    def loglike(self, x):
        yhat = []
        for i in range(self.n_classes):
            ll = self.classifiers[i].loglike(x)
            yhat.append(ll)
        return np.array(yhat)


(x_train, y_train), (x_test, y_test) = get_mnist("data/").load_data()
# classifier = gaussian_classifier()
# classifier.train(x_test, y_test)
# r = classifier.loglike(x_test[0])
# print(r)

