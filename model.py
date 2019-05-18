import tensorflow as tf
import numpy as np
import os

import ops
import optim
from ops import squeeze2d, unsqueeze2d, linear_zeros
from layers import revnet2d, split2d, split2d_reverse, prior


def codec(inputs, objective, hps, reverse, cond, eps=None, reuse=None):
    """
    Multi-scale architecture
    """
    levels = range(hps.num_levels) if not reverse else reversed(range(hps.num_levels))
    epsilons = []
    for level in levels:
        if reverse and level < hps.num_levels - 1:
            inputs = split2d_reverse(inputs, eps=eps[level], hps=hps, name="pool"+str(level))

        inputs, objective = revnet2d(inputs, cond=cond, logdet=objective, reverse=reverse,
                                     hps=hps, name="flow_%s" % str(level), reuse=reuse)

        if not reverse and level < hps.num_levels - 1:
            inputs, objective, eps = split2d(inputs, objective=objective,
                                             hps=hps, name="pool"+str(level))
            epsilons.append(eps)

    if not reverse:
        return inputs, objective, epsilons
    else:
        return inputs, objective


class Model(object):
    def __init__(self, hps):
        self.hps = hps
        self.num_classes = hps.num_classes
        self.num_bins = 2**hps.num_bits_x

        self.global_step = tf.Variable(1, dtype=tf.float32, trainable=False, name='create_inputs/global_step')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

    def encode(self, inputs, labels, condition=None):
        ## Dequantization by adding uniform noise
        with tf.variable_scope("preprocess"):
            self.y = tf.one_hot(labels, depth=self.num_classes, dtype=tf.float32)

            inputs = tf.cast(inputs, 'float32')
            self.height, self.width, self.channels = inputs.get_shape().as_list()[1:]
            if self.hps.num_bits_x < 8:
                inputs = tf.floor(inputs/2**(8-self.hps.num_bits_x))
            inputs = inputs / self.num_bins - 0.5
            inputs = inputs + tf.random_uniform(tf.shape(inputs), 0, 1./self.num_bins)

            objective = tf.zeros(self.hps.batch_size)
            objective += -np.log(self.num_bins) * np.prod(ops.shape(inputs)[1:])
            inputs = squeeze2d(inputs)

        ## Encoder
        if self.hps.conditioning and condition is None:
            condition = self.y
            # with tf.variable_scope("cond_preprocess"):
            #     condition = tf.layers.dense(condition, units=10, use_bias=False)
        z, objective, eps = codec(inputs, cond=condition, objective=objective, hps=self.hps, reverse=False)

        ## Prior
        with tf.variable_scope("prior"):
            self.hps.top_shape = z.get_shape().as_list()[1:]
            logp, sample, get_eps = prior(self.y, self.hps)
            obj = logp(z)
            eps.append(get_eps(z))
            objective += obj
            self.objective = -objective

            # Class label predict with latent representation
            if self.hps.ycond:
                z_y = tf.reduce_mean(z, axis=[1, 2])
                self.logits = linear_zeros(z_y, self.num_classes, name="classifier")
        return eps

    def decode(self, labels=None, condition=None, epsilon=None):
        """
        Args:
            labels: Class label, could be none
            condition: 2D or 4D tensor, condition for dynamic linear transformation
            epsilon: None or list. If specified, it should be a list with `num_levels` elements

        Returns:
            x: 4D tensor, generated samples
        """
        with tf.variable_scope("prior", reuse=tf.AUTO_REUSE):
            if labels is None:
                y_onehot = self.y
            elif len(labels.shape) == 1:
                y_onehot = tf.one_hot(labels, depth=self.num_classes, dtype=tf.float32)
            elif len(labels.shape) == 2:
                y_onehot = labels

            _, sample, get_eps = prior(y_onehot, self.hps)

            if epsilon is not None:
                eps = epsilon if len(epsilon) == self.hps.num_levels else [None] * (self.hps.num_levels-1) + epsilon
            else:
                eps = [None] * self.hps.num_levels

            z = sample(eps=eps[-1])
            objective = tf.zeros(tf.shape(z)[0])

        if self.hps.conditioning and condition is None:
            condition = y_onehot
            # with tf.variable_scope("cond_preprocess", reuse=tf.AUTO_REUSE):
            #     condition = tf.layers.dense(condition, units=10, use_bias=False)
        z, objective = codec(z, cond=condition, hps=self.hps, reverse=True,
                             objective=objective, eps=eps[:-1], reuse=tf.AUTO_REUSE)

        with tf.variable_scope("postprocess"):
            x = unsqueeze2d(z)
            x = tf.clip_by_value(tf.floor((x+.5)*self.num_bins)*(256./self.num_bins), 0, 255)
            self.gen_x = tf.cast(x, 'uint8')
        return self.gen_x

    def _loss(self):
        with tf.name_scope("loss"):
            bits_x = self.objective / (np.log(2.) * self.height * self.width * self.channels)
            if self.hps.weight_y > 0. and self.hps.ycond:
                bits_y = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)/np.log(2.)
                total_loss = tf.reduce_mean(bits_x + self.hps.weight_y * bits_y)
            else:
                bits_y = tf.zeros_like(bits_x)
                total_loss = tf.reduce_mean(bits_x)
            bits_x = tf.reduce_mean(bits_x)
            bits_y = tf.reduce_mean(bits_y)

            return total_loss, bits_x, bits_y

    def train(self, inputs, labels, condition=None, allreduce=None):
        with tf.variable_scope(tf.get_variable_scope()):
            tower_grads = []
            tower_loss = []
            bits_x = []
            bits_y = []
            for i in range(self.hps.num_gpus):
                start = i * self.hps.batch_size
                end = (i + 1) * self.hps.batch_size
                tower_output = self._single_tower(tower_idx=i,
                                                  inputs=inputs[start:end],
                                                  labels=labels[start:end],
                                                  condition=condition if condition is None else condition[start:end])
                tower_grads.append(tower_output[0])
                tower_loss.append(tower_output[1])
                bits_x.append(tower_output[2])
                bits_y.append(tower_output[3])

            grads = self._average_gradients(tower_grads)
            self.total_loss = tf.reduce_mean(tower_loss, name="loss/total_loss")
            self.bits_x = tf.reduce_mean(bits_x, name="loss/bits_x")
            self.bits_y = tf.reduce_mean(bits_y, name="loss/bits_y")

            all_params = tf.trainable_variables()
            self.num_variables = np.sum([np.prod(var.shape) for var in all_params])

        # polyak average parameters
        self.hps.train_its = 1000
        self.hps.polyak_epochs = 1

        optimizer = {'adam': optim.adam, 'adamax': optim.adamax}[self.hps.optimizer]
        self.train_ops, self.polyak_ops, _ = optimizer(all_params,
                                                       grads, alpha=self.lr,
                                                       global_step=self.global_step,
                                                       hps=self.hps,
                                                       allreduce=allreduce)
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)
        self.test_saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
        self.summary_ops = tf.summary.merge_all()

    def initialize(self, sess, ckpt_path=None):
        """
        Model initialization. Try to initiailize model from checkpoint if `ckpt_path`
        is specified.
        """
        epoch = step = 1
        # Randomly initialize model
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(ckpt_path)
        try:
            # Try to restore all trainable variables from checkpoint
            self.saver.restore(sess, ckpt)
            msg = "Restored global variables from: %s" % ckpt
            names = os.path.basename(ckpt).split("-")
            epoch += int(names[-2])
            step = int(names[-1])
        except:
            graph = tf.get_default_graph()
            try:
                # Try to restore trainable variables those can be found in checkpoint
                reader = tf.train.NewCheckpointReader(ckpt)
                var_list = reader.get_variable_to_shape_map().keys()
                var_list = [v for v in var_list if "adam" not in v and "MovingAverage" not in v]
                restore_list = []
                for v in var_list:
                    try:
                        restore_list.append(graph.get_tensor_by_name(v+":0"))
                    except:
                        continue
                saver = tf.train.Saver(restore_list)
                saver.restore(sess, ckpt)
                msg = "Restored trainable variables subset from: %s" % ckpt
                names = os.path.basename(ckpt).split("-")
                epoch += int(names[-2])
                step = int(names[-1])
            except:
                msg = "Train from scratch"
        return epoch, step, msg

    def save(self, sess, path, epoch, global_step):
        path = os.path.join(path, "model-%s" % epoch)
        self.saver.save(sess, path, global_step=global_step)

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        """
        with tf.name_scope("average_gradients"):
            average_grads = []
            for grads in zip(*tower_grads):
                grads = tf.stack(grads)
                grad = tf.reduce_mean(grads, 0)
                average_grads.append(grad)
            return average_grads

    def _single_tower(self, tower_idx, inputs, labels, condition):
        with tf.device('/gpu:%d' % tower_idx):
            self.encode(inputs, labels, condition)
            total_loss, bits_x, bits_y = self._loss()
            all_params = tf.trainable_variables()

            # L2 regularization for weights of invertible 1x1 convolutions
            # To solve the NaN issuse ('Input is not invertible' error)
            with tf.name_scope("l2_loss"):
                if self.hps.l2_factor > 0.:
                    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in all_params if "invconv/W" in v.name])
                    n = np.sum([np.prod(v.shape) for v in all_params if "invconv/W" in v.name])
                    l2_loss = (self.hps.l2_factor / tf.to_float(n)) * self.l2_loss
                    total_loss += l2_loss
                else:
                    self.l2_loss = tf.constant(0.)

            grads = tf.gradients(total_loss, all_params)
            tf.get_variable_scope().reuse_variables()

        return grads, total_loss, bits_x, bits_y
