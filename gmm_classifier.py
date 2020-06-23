from mnist_dataset import get_mnist
import pdb

(x_train, y_train), (x_test, y_test) = get_mnist("data/").load_data()

pdb.set_trace()