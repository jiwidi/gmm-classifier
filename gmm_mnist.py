from mnist_dataset import get_mnist
from gmm_classifier import gmm_classifier
import numpy as np
from sklearn.decomposition import PCA


def main():
    classifier = gmm_classifier()
    (x_train, y_train), (x_test, y_test) = get_mnist("data/").load_data()
    pca = PCA(n_components=30)
    pca.fit(x_train)
    x_train_tmp = pca.transform(x_train)
    x_test_tmp = pca.transform(x_test)
    classifier.train(x_train_tmp, y_train, x_test_tmp, y_test, 5, 0.9)

    # x_test = np.array(
    #     [
    #         [3, 4, 5],
    #         [5, 4, 3],
    #         [1, 1, 1],
    #         [3, 33, 5],
    #         [5, 235, 3],
    #         [1, 6, 1],
    #         [231, 4, 5],
    #         [5, 123, 3],
    #         [7, 1, 1],
    #         [3, 4, 12],
    #         [5, 22, 3],
    #         [1, 8, 1],
    #         [3, 4, 5],
    #         [5, 67, 3],
    #         [1, 1, 1],
    #     ]
    # )

    # y_test = np.array([0, 1, 1, 0, 2, 0, 1, 1, 0, 2, 2, 1, 1, 0, 1])
    # classifier.train(x_test, y_test, x_test, y_test, 1, 1.0)


if __name__ == "__main__":
    main()

