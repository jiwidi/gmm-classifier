import numpy as np
import gzip
import struct
from array import array
from os.path  import join

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []

        with gzip.open(labels_filepath, 'rb') as file:
            file.read(8) #
            c = file.read()
            labels = np.frombuffer(c, dtype=np.uint8).astype(np.int16)

        with gzip.open(images_filepath, 'rb') as file:
            file.read(16) #
            c = file.read()
            size = int(len(c)/28/28)
            image_data = np.frombuffer(c, dtype=np.uint8).astype(np.int16).reshape(size,28*28)

        return image_data, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

def get_mnist(input_path):
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte.gz')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte.gz')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte.gz')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte.gz')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    return mnist_dataloader
