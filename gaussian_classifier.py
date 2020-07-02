from mnist_dataset import get_mnist
import pdb
import numpy as np
import math


def logdet(X):
    lamb = np.linalg.eig(X)[0]
    if np.any(lamb <= 0.0):
        return np.log(2.2204e-16)
    else:
        return np.sum(np.log(lamb), axis=0)


class gaussian_classifier:
    def __init__(self):
        self.n_classes = 0
        self.pc = []
        self.mu = []
        self.sigma = []

    def train(self, x, y, alpha=1.0):
        self.n_classes = len(set(y))
        self.mu = []
        for c in range(self.n_classes):
            mask = np.where(y == c)[0]
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
        pxGc = qua + lin + cons
        return pxGc

    def predict(self, X):
        gte = np.zeros([self.n_classes, len(X)])
        for c in range(self.n_classes):
            gte[c] = np.log(self.pc[c]) + self.compute_pxGc(X, c)
        return np.argmax(gte, axis=0)

