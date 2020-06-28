from mnist_dataset import get_mnist
from gaussian_classifier import gaussian_classifier
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def main():
    (x_train, y_train), (x_test, y_test) = get_mnist("data/").load_data()

    # 4.1
    print("Gaussian classifier with mnist dataset reduced by PCA")
    pca_components = list(range(1, 15))
    pca_results = []
    for pca_component in pca_components:
        pca = PCA(n_components=pca_component)
        pca.fit(x_train)
        x_train_tmp = pca.transform(x_train)
        x_test_tmp = pca.transform(x_test)
        gauss = gaussian_classifier()
        gauss.train(x_train_tmp, y_train)
        yhat = gauss.predict(x_test_tmp)
        pca_results.append(np.mean(y_test != yhat) * 100)
    ##Save plot
    fig, ax = plt.subplots()
    plt.title("pca components vs error rate in gaussian classifier")
    plt.plot(pca_components, pca_results, label="Error rate on test set alpha=1.0")
    plt.legend(loc="upper left")
    plt.xticks(pca_components, rotation=70)
    plt.xlabel("PCA Dimensions")
    plt.ylabel("Eror rate")
    plt.savefig("scripts/results/gc_pca_4-1.png")
    # 4.2
    print("Gaussian classifier with mnist dataset reduced by PCA with smoothing")
    alphas = [0.01, 0.5, 0.9]
    plt.title("pca components vs error rate in gaussian classifier with smoothing")
    plt.plot(pca_components, pca_results)
    for alpha in alphas:
        pca__smooth_results = []
        for pca_component in pca_components:
            pca = PCA(n_components=pca_component)
            pca.fit(x_train)
            x_train_tmp = pca.transform(x_train)
            x_test_tmp = pca.transform(x_test)
            gauss = gaussian_classifier()
            gauss.train(x_train_tmp, y_train, alpha=alpha)
            yhat = gauss.predict(x_test_tmp)
            pca__smooth_results.append(np.mean(y_test != yhat) * 100)
        plt.plot(pca_components, pca__smooth_results, marker="o", label=f"Error rate on test set alpha={alpha}")
    ##Save plot
    plt.legend(loc="upper left")
    plt.xlabel("PCA Dimensions")
    plt.ylabel("Eror rate")
    plt.xticks(pca_components, rotation=70)
    plt.savefig("scripts/results/gc_pca_4-2.png")


if __name__ == "__main__":
    main()
