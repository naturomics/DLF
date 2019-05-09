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


def load_cifar10(path, split):
    split = split.lower()
    if split == 'eval':
        fpath = os.path.join(path, 'cifar-10-batches-py', 'test_batch')
        images, labels = load_batch(fpath)
    else:
        num_samples = 50000
        images = np.empty((num_samples, 3, 32, 32), dtype='uint8')
        labels = np.empty((num_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = os.path.join(path, 'cifar-10-batches-py', 'data_batch_' + str(i))
            (images[(i - 1) * 10000:i * 10000, :, :, :],
             labels[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    images = np.reshape(images.transpose(0, 2, 3, 1), (-1, 3072)).astype(np.float32)
    labels = np.reshape(labels, (-1, )).astype(np.int32)

    return(zip(images, labels))


class DataLoader(DataLoader):
    def __init__(self, path=None,
                 num_worker=1,
                 one_hot=False,
                 name=None, **kwargs):
        self.height = self.width = 32
        self.channels = 3
        self.num_classes = 10
        self.batched_shapes = {'images': [-1, self.height, self.width, self.channels],
                               'labels': [-1]}

        if path is None:
            path = os.path.join('data', 'cifar10')
            os.makedirs(path, exist_ok=True)
        def downloader(path):
            maybe_download_and_extract("cifar-10-python", path)
            return path
        super(DataLoader, self).__init__(path=path,
                                         num_worker=num_worker,
                                         one_hot=one_hot,
                                         downloader=downloader,
                                         name=name, **kwargs)

    def tfrecorder(self, path):
        train_set = load_cifar10(path, 'train')
        eval_set = load_cifar10(path, 'eval')

        train_set_outpath = os.path.join(path, "train_cifar10.tfrecords")
        eval_set_outpath = os.path.join(path, "validation_cifar10.tfrecords")

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
