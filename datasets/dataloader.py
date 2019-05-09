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
from glob import glob
import tensorflow as tf
from abc import abstractmethod

class DataLoader(object):
    def __init__(self, path=None,
                 shards=1,
                 rank = None,
                 one_hot=False,
                 downloader=None,
                 threads_fmap=2,
                 threads_dmap=4,
                 name=None,
                 buffer_size=50000,
                 **kwargs):
        """
        Args:
            path: Path where store data
            one_hot: Boolean, return one-hoted labels if true.
            downloader: None or a function with parameter 'path' and return a new 'path'.
            name: Name.
        """

        if hasattr(downloader, "__call__"):
            path = os.path.join(os.environ["HOME"], "DLF", "data") if path is None else path
            self.path = downloader(path)
            assert self.path is not None
        elif path is not None:
            self.path = path
        else:
            raise ValueError("Define your data downloader function or give a path where store data")

        if not glob(os.path.join(self.path, "*.tfrecords"), recursive=True):
            self.path = self.tfrecorder(self.path)
            if self.path is None:
                raise ValueError("Bad tfrecorder implementation, returned 'path' is None.")
            elif not glob(os.path.join(self.path, "*.tfrecords")):
                raise ValueError("Bad tfrecorder implementation, no tfrecord files generated.")

        self.buffer_size = buffer_size
        self.shards = shards
        self.rank = rank
        name = "create_inputs/handle" if name is None else name + "/handle"
        self.handle = tf.placeholder(tf.string, shape=[], name=name)
        self.next_element = None
        self.name = name
        self.threads_fmap = threads_fmap
        self.threads_dmap = threads_dmap

    def __call__(self, batch_size, mode):
        """
        Args:
            batch_size: Integer.
            mode: Running phase, one of "train", or "eval".
        """
        with tf.name_scope(self.name, "create_inputs"):
            mode = mode.lower()
            modes = ["train", "eval"]
            if mode == "train":
                file_pattern = os.path.join(self.path, "*%s*.tfrecords" % mode)
            elif mode == "eval" or mode == "test":
                file_pattern = os.path.join(self.path, "*validation*.tfrecords")
            else:
                raise ValueError("mode not found! supported modes are " + modes)

            files = tf.data.Dataset.list_files(file_pattern,
                                               shuffle=True if mode == "train" else False)
            if self.rank is not None:
                files = files.shard(self.shards, self.rank)

            dataset = files.apply(tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=self.threads_fmap))
            if mode == "train":
                dataset = dataset.shuffle(buffer_size=16*batch_size)
            dataset = dataset.repeat(1)
            dataset = dataset.apply(tf.data.experimental.map_and_batch(self.parser,
                                                                       batch_size=batch_size,
                                                                       num_parallel_calls=self.threads_dmap))
            dataset = dataset.prefetch(buffer_size=2*batch_size)
            iterator = dataset.make_initializable_iterator()

            if self.next_element is None:
                self.next_element = tf.data.Iterator.from_string_handle(self.handle,
                                                                        iterator.output_types,
                                                                        iterator.output_shapes).get_next()
            return iterator

    def initialize(self, sess, iterator, get_handle=False):
        sess.run(iterator.initializer)
        if get_handle:
            handle = sess.run(iterator.string_handle())
            return handle

    @abstractmethod
    def tfrecorder(self, path):
        """ Function for converting dataset to tfrecord files.

        Args:
            path: Path where stores dataset.

        Return:
            path: Path where stores the output tfrecord files.
        """
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def parser(self, serialized_example):
        """ Data parsing function.
        """
        raise NotImplementedError('Not implemented')

    def get_element(self, *args):
        if len(args) == 1:
            return tf.reshape(self.next_element[args[0]],
                               self.batched_shapes[args[0]],
                               name=self.name+"/reshape_%s" % args[0])
        elif len(args) > 1:
            return [tf.reshape(self.next_element[name],
                               self.batched_shapes[name],
                               name=self.name+"/reshape_%s" % name) for name in args]
        else:
            raise ValueError("Expected at least one input, got None")
