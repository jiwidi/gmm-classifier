from mnist_dataset import get_mnist
import pdb
import numpy as np
import math


def log_gaussian(o, mu, r):
    compute = (-0.5 * np.log(r) - np.divide(np.square(o - mu), 2 * r) - 0.5 * np.log(2 * math.pi)).sum()
    return compute


def logdet(X):
    lamb = np.linalg.eig(X)[0]
    if np.any(lamb <= 0.0):
        return np.log(2.2204e-16)
    else:
        return np.sum(np.log(lamb), axis=0)


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
        self.n_classes = 0
        self.pc = []
        self.mu = []
        self.sigma = []

    def train(self, x, y, alpha=1.0):
        self.n_classes = len(set(y))
        self.mu = []
        for i in range(self.n_classes):
            mask = np.where(y == i)[0]
            xtmp = x[mask]
            ytmp = y[mask]
            self.pc.append(len(xtmp) / len(x))  # Probability of class C
            muc = np.sum(xtmp, axis=0) / len(xtmp)
            self.mu.append(muc)
            # sigma = np.sum(np.matmul(xtmp - muc, xtmp - np.transpose(muc)), axis=0) Why not this?
            sigma = ((xtmp - muc).T.dot(xtmp - muc)) / len(xtmp)
            # Smoothing
            sigma = alpha * sigma + (1 - alpha) * np.eye(x.shape[1])
            self.sigma.append(sigma)

    def compute_pxGc(self, X, ic):
        sigma = self.sigma[ic]
        mu = self.mu[ic]
        qua = -0.5 * np.sum(np.multiply(np.matmul(X, np.linalg.pinv(sigma)), X), axis=1)
        lin = X.dot(mu.T.dot(np.linalg.pinv(sigma))).T
        cons = -0.5 * logdet(sigma)
        cons = cons - 0.5 * (mu.T.dot(np.linalg.pinv(sigma)).dot(mu))
        return qua + lin + cons

    def predict(self, X):
        gte = np.zeros([self.n_classes, len(X)])
        for i in range(self.n_classes):
            gte[i] = np.log(self.pc[i]) + self.compute_pxGc(X, i)
        return np.argmax(gte, axis=0)


(x_train, y_train), (x_test, y_test) = get_mnist("data/").load_data()
print(x_test.shape)
classifier = gaussian_classifier()
# x_test = np.array([[3, 4, 5], [5, 4, 3], [1, 1, 1]])
# y_test = np.array([0, 1, 1])
classifier.train(x_train, y_train)
r = classifier.predict(x_test)
# r = classifier.compute_pxGc(x_test, 1)
print(r[:20])
print(y_test[:20])

