from mnist_dataset import get_mnist
import pdb
import numpy as np
import math

np.random.seed(23)


def log_gaussian(o, mu, r):
    compute = (-0.5 * np.log(r) - np.divide(np.square(o - mu), 2 * r) - 0.5 * np.log(2 * math.pi)).sum()
    return compute


def logdet(X):
    np.savetxt("test.out", X)
    return np.linalg.slogdet(X)[1]
    lamb = np.linalg.eig(X)[0]
    if np.any(lamb <= 0.0):
        return np.log(2.2204e-16)
    else:
        print("det")
        return np.sum(np.log(lamb), axis=0)


class gmm_classifier:
    def __init__(self):
        self.n_classes = 0
        self.pc = []
        self.mu = []
        self.sigma = []
        self.pkGc = []

    def compute_zk(self, X, ic, k):
        mu = self.mu[ic][:, k]
        sigma = self.sigma[ic][k]
        D = X.shape[1]
        cons = np.log(self.pkGc[ic][k])
        cons = cons - 0.5 * D * np.log(2 * np.pi)
        cons = cons - 0.5 * logdet(sigma)
        cons = cons - 0.5 * mu.T.dot(np.linalg.pinv(sigma).dot(mu))
        lin = X.dot(mu.T.dot(np.linalg.pinv(sigma)).T)
        qua = -0.5 * np.sum(np.multiply(X.dot(np.linalg.pinv(sigma)), X), axis=1)
        zk = qua + lin + cons
        return zk

    def train(self, x, y, xt, yt, K, alpha=1.0):
        self.n_classes = len(set(y))
        self.sigma = [[] for _ in range(self.n_classes)]
        # Initialization of parameters for the mixture of gaussians
        # Estimation of class priors as histogram counts
        pc = np.bincount(y) / x.shape[0]
        for c in range(self.n_classes):
            # Initialization of component priors p(k|c) as uniform distro
            self.pkGc.append([1 / K for _ in range(K)])
            mask = np.where(y == c)[0]
            xtmp = x[mask]
            # Initialization of K component means mu_kc as K random samples from class c
            choices = np.random.choice(mask, K)
            muc = x[choices].T
            self.mu.append(muc)
            #  Initialization of K component covariance sigma_kc
            #  as K class covariance matrix divided by the number of components K
            sigma = np.cov(xtmp, rowvar=False, bias=True) / K
            self.sigma[c] = [sigma for _ in range(K)]
        epsilon = 1e-4
        L = float("-inf")
        it = 0
        oL = L
        while not ((L - oL) / abs(oL) < epsilon):
            oL = L
            L = 0
            for c in range(self.n_classes):
                # E step
                mask = np.where(y == c)[0]
                xtmp = x[mask]
                Nc = len(xtmp)
                zk = []
                for k in range(K):
                    zk.append(self.compute_zk(xtmp, c, k))
                zk = np.array(zk).T
                # Robust computation of znk and log-likehood
                maxzk = zk.max(axis=1)
                zk = np.exp(zk - maxzk[:, None])
                sumzk = np.sum(zk, 1)
                zk = np.divide(zk, sumzk[:, None])
                L = L + Nc * np.log(pc[c]) + np.sum(maxzk + np.log(sumzk), 0)
                # M step: parameter update
                # Weight of each component
                sumzk = np.sum(zk, 0)
                self.pkGc[c] = sumzk / Nc
                self.mu[c] = np.divide(xtmp.T.dot(zk), sumzk)
                for k in range(K):
                    covar = (
                        (xtmp - self.mu[c][:, k].T).T.dot(np.multiply(xtmp - self.mu[c][:, k].T, zk[:, k][:, None]))
                    ) / sumzk[k]
                    # Smoothing
                    # ((a - a[:, k].T).T.dot(np.multiply(a - a[:, k].T, zk[:, k])))
                    self.sigma[c][k] = alpha * covar + (1 - alpha) * np.eye(x.shape[1])
            # Likelihood divided by the number of training samples
            L = L / len(x)
            # Compute g for training and test sets
            gtr = []
            gte = []
            for c in range(self.n_classes):
                # Training set
                zk = []
                for k in range(K):
                    zk.append(self.compute_zk(x, c, k))
                zk = np.array(zk).T
                # Robust computation of znk
                maxzk = zk.max(axis=1)
                zk = np.exp(zk - maxzk[:, None])
                sumzk = np.sum(zk, 1)
                tmp_gtr = np.log(pc[c]) + maxzk + np.log(sumzk)
                gtr.append(tmp_gtr)
                # Test set
                zk = []
                for k in range(K):
                    zk.append(self.compute_zk(xt, c, k))
                zk = np.array(zk).T
                # Robust computation of znk
                maxzk = zk.max(axis=1)
                zk = np.exp(zk - maxzk[:, None])
                sumzk = np.sum(zk, 1)
                tmp_gte = np.log(pc[c]) + maxzk + np.log(sumzk)
                gte.append(tmp_gte)
            gtr = np.array(gtr).T
            gte = np.array(gte).T
            # Classification of training and test sets and error estimation
            ## Training
            yhat = np.argmax(gtr, axis=1)
            trerr = np.mean(y != yhat) * 100
            ## Test
            yhat = np.argmax(gte, axis=1)
            teerr = np.mean(y_test != yhat) * 100
            it += 1
            print("{:4} {:11} {:11} {:5} {:5}".format(it, oL, L, trerr, teerr))

        # print(len(self.sigma[0][0]))

    def predict(self, X):
        gte = np.zeros([self.n_classes, len(X)])
        for i in range(self.n_classes):
            gte[i] = np.log(self.pc[i]) + self.compute_pxGc(X, i)
        return np.argmax(gte, axis=0)


# (x_train, y_train), (x_test, y_test) = get_mnist("data/").load_data()
classifier = gmm_classifier()
x_test = np.array(
    [
        [3, 4, 5],
        [5, 4, 3],
        [1, 1, 1],
        [3, 33, 5],
        [5, 235, 3],
        [1, 6, 1],
        [231, 4, 5],
        [5, 123, 3],
        [7, 1, 1],
        [3, 4, 12],
        [5, 22, 3],
        [1, 8, 1],
        [3, 4, 5],
        [5, 67, 3],
        [1, 1, 1],
    ]
)

y_test = np.array([0, 1, 1, 0, 2, 0, 1, 1, 0, 2, 2, 1, 1, 0, 1])
classifier.train(x_test, y_test, x_test, y_test, 2, 1.0)
r = classifier.predict(x_test)
acc = np.mean(y_test == r) * 100
# r = classifier.compute_pxGc(x_test, 1)
print(r[:20])
print(y_test[:20])
print(100 - acc)

