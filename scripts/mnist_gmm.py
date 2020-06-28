from mnist_dataset import get_mnist
from gmm_classifier import gmm_classifier
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def main():
    classifier = gmm_classifier()
    (x_train, y_train), (x_test, y_test) = get_mnist("data/").load_data()
    # 5.2
    pca_components = [u for u in range(2, 30)]
    pca_results = []
    print("GMM with mnist dataset reduced by PCA with smoothing")
    alphas = [0.1, 0.5, 0.9]
    for alpha in alphas:
        pca__smooth_results = []
        for pca_component in pca_components:
            pca = PCA(n_components=pca_component)
            pca.fit(x_train)
            x_train_tmp = pca.transform(x_train)
            x_test_tmp = pca.transform(x_test)
            gauss = gmm_classifier()
            gauss.train(x_train_tmp, y_train, x_test_tmp, y_test, 1, alpha=alpha)
            yhat = gauss.predict(x_test_tmp)
            pca__smooth_results.append(np.mean(y_test != yhat) * 100)
        plt.plot(pca_components, pca__smooth_results, marker="o", label=f"Error rate on test set with alpha={alpha} and k=1")
    ##Save plot
    plt.title("pca components vs error rate in GMM - K=1, Alpha=0.9")
    plt.legend(loc="upper left")
    plt.xticks(pca_components, rotation=70)
    plt.xlabel("PCA Dimensions")
    plt.ylabel("Eror rate")
    plt.savefig("scripts/results/gmm_pca_5-2.png")
    # 5.3
    plt.clf()
    pca_results = []
    print("GMM with mnist dataset reduced by PCA with smoothing")
    ks = [1, 2, 3, 4, 5, 6, 7]
    pca__smooth_results = []
    for k in ks:
        pca = PCA(n_components=30)
        pca.fit(x_train)
        x_train_tmp = pca.transform(x_train)
        x_test_tmp = pca.transform(x_test)
        gauss = gmm_classifier()
        gauss.train(x_train_tmp, y_train, x_test_tmp, y_test, k, alpha=0.9)
        yhat = gauss.predict(x_test_tmp)
        pca__smooth_results.append(np.mean(y_test != yhat) * 100)
    plt.plot(ks, pca__smooth_results, marker="o", label=f"Error rate on test set with alpha={alpha} and k={k}")
    ##Save plot
    plt.title("Number of mixtures vs error rate in GMM - Alpha=0.9 - PCA to 30dim")
    plt.legend(loc="upper left")
    plt.xticks(ks, rotation=70)
    plt.xlabel("Number of mixtures")
    plt.ylabel("Eror rate")
    plt.savefig("scripts/results/gmm_pca_5-3.png")


if __name__ == "__main__":
    main()

