import os
import sys
import gzip
import shutil
import tarfile
import urllib
import tensorflow as tf


# mnist dataset
HOMEPAGE = "http://yann.lecun.com/exdb/mnist/"
MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"

# fashion-mnist dataset
HOMEPAGE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
FASHION_MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
FASHION_MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
FASHION_MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"

# CIFAR-10 dataset
HOMEPAGE = "http://www.cs.toronto.edu/~kriz/"
CIFAR_10_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_10_BIN_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"


def maybe_download_and_extract(dataset, save_to, force=False):
    if dataset in ['mnist', 'MNIST']:
        download_and_uncompress_zip(MNIST_TRAIN_IMGS_URL, save_to, force, dataset)
        download_and_uncompress_zip(MNIST_TRAIN_LABELS_URL, save_to, force, dataset)
        download_and_uncompress_zip(MNIST_TEST_IMGS_URL, save_to, force, dataset)
        download_and_uncompress_zip(MNIST_TEST_LABELS_URL, save_to, force, dataset)
    elif dataset in ['fashion-mnist', 'fashion_mnist', 'fashionmnist', 'fashionMNIST']:
        download_and_uncompress_zip(FASHION_MNIST_TRAIN_IMGS_URL, save_to, force, dataset)
        download_and_uncompress_zip(FASHION_MNIST_TRAIN_LABELS_URL, save_to, force, dataset)
        download_and_uncompress_zip(FASHION_MNIST_TEST_IMGS_URL, save_to, force, dataset)
        download_and_uncompress_zip(FASHION_MNIST_TEST_LABELS_URL, save_to, force, dataset)
    elif dataset in ['cifar-10-bin', 'cifar10bin', "cifar_10_bin"]:
        download_and_uncompress_zip(CIFAR_10_BIN_URL, save_to, force, dataset)
    elif dataset in ['cifar-10-python', 'cifar10python', "cifar_10_python"]:
        download_and_uncompress_zip(CIFAR_10_URL, save_to, force, dataset)
    else:
        raise Exception("Invalid dataset name! Got: ", dataset)


def download_and_uncompress_zip(URL, dataset_dir, force=False, dataset=None):
    '''
    Args:
        URL: the download links for data
        dataset_dir: the path to save data
        force: redownload data
    '''
    filename = URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    extract_to = os.path.splitext(filepath)[0]

    def download_progress(count, block_size, total_size):
        sys.stdout.write("\r>> (%s)Downloading %s %.1f%%" % (dataset, filename, float(count * block_size) / float(total_size) * 100.))
        sys.stdout.flush()

    if not force and os.path.exists(filepath):
        pass
    else:
        filepath, _ = urllib.request.urlretrieve(URL, filepath, download_progress)
        print()
        print('Successfully Downloaded', filename)

    if filepath.endswith(".tar.gz"):
        tar = tarfile.open(filepath, "r:gz")
        tar.extractall(dataset_dir)
        tar.close()
        return 0
    with gzip.open(filepath, 'rb') as f_in, open(extract_to, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def int64_feature(value):
    """Casts value to a TensorFlow int64 feature list."""
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """Casts value to a TensorFlow bytes feature list."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
