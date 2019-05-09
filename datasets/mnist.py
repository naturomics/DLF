# Copyright 2018 The CapsLayer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from datasets.dataloader import DataLoader
import tensorflow as tf
from datasets.utils import maybe_download_and_extract
from datasets.utils import int64_feature, bytes_feature
from tensorflow.python.keras.datasets.cifar import load_batch


MNIST_FILES = {
    'train': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
    'eval': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
}

def load_mnist(path, split):
    split = split.lower()
    image_file, label_file = [os.path.join(path, file_name) for file_name in MNIST_FILES[split]]

    with open(image_file) as fd:
        images = np.fromfile(file=fd, dtype=np.uint8)
        images = images[16:].reshape(-1, 784).astype(np.float32)
    with open(label_file) as fd:
        labels = np.fromfile(file=fd, dtype=np.uint8)
        labels = labels[8:].astype(np.int32)
    return(zip(images, labels))


class DataLoader(DataLoader):
    def __init__(self, path=None,
                 num_worker=1,
                 one_hot=False,
                 name=None, **kwargs):
        self.height = self.width = 28
        self.channels = 1
        self.num_classes = 10
        self.batched_shapes = {'images': [-1, self.height, self.width, self.channels],
                               'labels': [-1]}

        if path is None:
            path = os.path.join('data', 'mnist')
            os.makedirs(path, exist_ok=True)
        def downloader(path):
            maybe_download_and_extract("mnist", path)
            return path
        super(DataLoader, self).__init__(path=path,
                                         num_worker=num_worker,
                                         one_hot=one_hot,
                                         downloader=downloader,
                                         name=name, **kwargs)

    def tfrecorder(self, path):
        train_set = load_mnist(path, 'train')
        eval_set = load_mnist(path, 'eval')

        train_set_outpath = os.path.join(path, "train_mnist.tfrecords")
        eval_set_outpath = os.path.join(path, "validation_mnist.tfrecords")

        def encode_and_write(dataset, filename):
            with tf.python_io.TFRecordWriter(filename) as writer:
                for image, label in dataset:
                    image_raw = image.tostring()
                    example = tf.train.Example(features=tf.train.Features(
                        feature={'image': bytes_feature(image_raw),
                                 'label': int64_feature(label)}))
                    writer.write(example.SerializeToString())

        if not os.path.exists(train_set_outpath):
            encode_and_write(train_set, train_set_outpath)
        if not os.path.exists(eval_set_outpath):
            encode_and_write(eval_set, eval_set_outpath)
        return path

    def parser(self, serialized_record):
        features = tf.parse_single_example(serialized_record,
                                           features={'image': tf.FixedLenFeature([], tf.string),
                                                     'label': tf.FixedLenFeature([], tf.int64)})
        image = tf.decode_raw(features['image'], tf.float32)
        image = tf.cast(image, tf.float32)
        label = tf.cast(features['label'], tf.int32)
        features = {'images': image, 'labels': label}
        return(features)
