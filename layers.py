from math import ceil, floor
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope

import ops
from ops import actnorm, invertible_1x1_conv, conv2d, conv2d_zeros, linear_zeros, linear
from ops import gaussian_diag, unsqueeze2d, squeeze2d


def revnet2d(inputs, cond, logdet, hps, reverse=False, reuse=False, name=None):
    """
    Stacks of dynamic linear transformations on the same scale.

    Args:
        inputs: 4D tensor, [batch_size, height, width, in_channels]
        cond: 2D or 4D conditions
        logdet: 1D tensor, objective

    Returns:
        output: 4D tensor with same shape of inputs
        logdet: objective
    """
    with tf.variable_scope(name, "revnet2d", reuse=reuse):
        seq = reversed(range(hps.depth)) if reverse else range(hps.depth)
        for i in seq:
            with tf.variable_scope("block_%s"%str(i)) as scope:
                inputs, logdet = revnet2d_step(inputs, cond=cond,
                                               logdet=logdet, hps=hps, reverse=reverse)
    return inputs, logdet


def revnet2d_step(inputs, cond, logdet, hps, reverse):
    """
    A step of flow, invertible 1x1 convolution followed by a layer of (inverse) dynamic linear transformation

    Args:
        inputs: 4D tensor
        cond: 2D or 4D tensor, conditions of s,mu = g(x, h)
        logdet: objective

    Returns:
        output: 4D tensor
        logdet: objective
    """
    channels = inputs.get_shape().as_list()[-1]
    assert channels % hps.num_parts == 0
    if hps.num_parts == 4 and hps.splitting == 1: # incremental sequence
        size_splits = [floor(channels/8), ceil(channels/8), channels//4, channels//2]
    elif hps.num_parts == 4 and hps.splitting == 2: # decrement sequence
        size_splits = [channels//2, channels//4, ceil(channels/8), floor(channels/8)]
    else:
        size_splits = [channels//hps.num_parts] * hps.num_parts

    if not reverse:
        inputs, logdet = invertible_1x1_conv(inputs, logdet=logdet, reverse=reverse,
                                             use_bias=hps.invconv_bias, name="invconv")

        with tf.variable_scope("dynamic_linear") as scope:
            xs = tf.split(inputs, num_or_size_splits=size_splits, axis=-1)

            logs = []
            shift = []
            shift_1, logs_1 = g_0(cond, shape=ops.shape(xs[1]), name="g_0")
            logs.append(logs_1)
            # dynamic linear transform
            if hps.decomposition == 0:
                shift.append(shift_1)
                for k_minus_1 in range(len(xs)-1):
                    shift_k, logs_k = g_k(xs[k_minus_1], cond=cond, hps=hps, filters=hps.width,
                                          channels=xs[k_minus_1+1].get_shape().as_list()[-1],
                                          name="g_%s" % str(k_minus_1+1))
                    shift.append(shift_k)
                    logs.append(logs_k)

                inputs += tf.concat(shift, axis=-1)
                logs = tf.concat(logs, axis=-1)
                inputs *= tf.exp(logs)
            # inverse dynamic linear transform
            elif hps.decomposition == 1:
                y_k_minus_1 = xs[0] + shift_1
                y_k_minus_1 *= tf.exp(logs_1)
                inputs = [y_k_minus_1]
                for k in range(1, len(xs)):
                    shift_k, logs_k = g_k(y_k_minus_1, cond=cond, hps=hps, filters=hps.width,
                                          channels=xs[k].get_shape().as_list()[-1],
                                          name="g_%s" % str(k))
                    y_k_minus_1 = xs[k] + shift_k
                    y_k_minus_1 *= tf.exp(logs_k)
                    inputs.append(y_k_minus_1)
                    logs.append(logs_k)
                inputs = tf.concat(inputs, axis=-1)
                logs = tf.concat(logs, axis=-1)
            else:
                raise ValueError("Decomposition type should be 0 or 1, got %s" % hps.decomposition)

            obj = tf.reduce_sum(logs, axis=[1, 2, 3])
            logdet += obj
    else:
        with tf.variable_scope("dynamic_linear") as scope:
            ys = tf.split(inputs, num_or_size_splits=size_splits, axis=-1)

            logs = []
            shift = []
            shift_1, logs_1 = g_0(cond, shape=ops.shape(ys[1]), name="g_0")
            logs.append(logs_1)
            # dynamic linear transform
            if hps.decomposition == 0:
                x_k_minus_1 = ys[0] * tf.exp(-logs_1)
                x_k_minus_1 -= shift_1
                inputs = [x_k_minus_1]
                for k in range(1, len(ys)):
                    shift_k, logs_k = g_k(x_k_minus_1, cond=cond, hps=hps, filters=hps.width,
                                          channels=ys[k].get_shape().as_list()[-1],
                                          name="g_%s" % str(k))
                    x_k_minus_1 = ys[k] * tf.exp(-logs_k)
                    x_k_minus_1 -= shift_k
                    inputs.append(x_k_minus_1)
                    logs.append(logs_k)
                inputs = tf.concat(inputs, axis=-1)
                logs = tf.concat(logs, axis=-1)
            # inverse dynamic linear transform
            elif hps.decomposition == 1:
                shift.append(shift_1)
                for k_minus_1 in range(len(ys)-1):
                    shift_k, logs_k = g_k(ys[k_minus_1], cond=cond, hps=hps, filters=hps.width,
                                          channels=ys[k_minus_1+1].get_shape().as_list()[-1],
                                          name="g_%s" % str(k_minus_1+1))
                    shift.append(shift_k)
                    logs.append(logs_k)
                logs = tf.concat(logs, axis=-1)
                inputs *= tf.exp(-logs)
                inputs -= tf.concat(shift, axis=-1)
            else:
                raise ValueError("Decomposition type should be 0 or 1, got %s" % hps.decomposition)

            obj = tf.reduce_sum(logs, axis=[1, 2, 3])
            logdet -= obj

        inputs, logdet = invertible_1x1_conv(inputs, logdet=logdet, reverse=reverse,
                                             use_bias=hps.invconv_bias, name="invconv")
    return inputs, logdet


def g_0(cond, shape, name=None):
    with tf.variable_scope(name, "g"):
        channels = shape[-1]
        inputs = tf.get_variable("h", [1, 1, 1, 2 * channels])
        inputs = inputs + tf.zeros(shape[:-1] + [2 * channels]) # broadcasting

        if cond is not None:
            rank = len(cond.shape)
            if rank == 2:
                inputs += tf.reshape(tf.layers.dense(cond, 2 * channels, use_bias=False),
                                shape=[-1, 1, 1, 2 * channels])
            elif rank == 4:
                inputs += conv2d(cond, width=2*channels, name="conv2d")
        shift = inputs[:, :, :, 0::2]
        logs = inputs[:, :, :, 1::2]

        # logs = alpha*tanh(logs)+beta, helpful for training stability
        rescale = tf.get_variable("rescale", [], initializer=tf.constant_initializer(1.))
        scale_shift = tf.get_variable("scale_shift", [], initializer=tf.constant_initializer(-3.))
        logs = tf.tanh(logs) * rescale + scale_shift

        return shift, logs


def g_k(inputs, cond, filters, hps, channels, reuse=None, name=None):
    """
    Three convolution layers for getting s_k,mu_k conditioning with x_{k-1} and condition h (if specified)

    Args:
        filters: the output channels of the first two convolution layers
        channels: the output channels of s_k, mu_k

    Returns:
        shift, logs: 4D tensor
    """
    with tf.variable_scope(name, "g_1", reuse=reuse):
        inputs = convnet(inputs, cond, filters, hps, channels=2 * channels)

        if cond is not None:
            rank = len(cond.shape)
            if rank == 2:
                inputs += tf.reshape(tf.layers.dense(cond, 2 * channels, use_bias=False),
                                shape=[-1, 1, 1, 2 * channels])
            elif rank == 4:
                inputs += conv2d(cond, width=2*channels, name="conv2d")

        shift = inputs[:, :, :, 0::2]
        logs = inputs[:, :, :, 1::2]

        # logs = alpha*tanh(logs)+beta, helpful for training stability
        rescale = tf.get_variable("rescale", [], initializer=tf.constant_initializer(1.))
        scale_shift = tf.get_variable("scale_shift", [], initializer=tf.constant_initializer(-3.))
        logs = tf.tanh(logs) * rescale + scale_shift

    return shift, logs


@add_arg_scope
def split2d(z, objective=0., hps=None, name=None):
    with tf.variable_scope(name):
        n_z = z.get_shape()[3]
        z1, z2 = tf.split(z, 2, axis=-1)
        pz = split2d_prior(z1, hps=hps)
        obj = pz.logp(z2)
        objective += obj
        z1 = squeeze2d(z1)
        eps = pz.get_eps(z2)
        return z1, objective, eps


@add_arg_scope
def split2d_reverse(z, eps, hps=None, name=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        z1 = unsqueeze2d(z)
        pz = split2d_prior(z1, hps=hps)
        z2 = pz.sample(eps=eps)
        z = tf.concat([z1, z2], 3)
        return z


def convnet(inputs, cond, filters, hps, channels):
    inputs = tf.nn.relu(conv2d(inputs, width=filters, name="conv2d_1"))
    inputs = tf.nn.relu(conv2d(inputs, filters, filter_size=[1,1], name="conv2d_2"))
    inputs = conv2d_zeros(inputs, channels, name="conv2d_3")
    return inputs


@add_arg_scope
def split2d_prior(z, hps):
    n_z2 = int(z.get_shape()[3])
    n_z1 = n_z2

    h = conv2d_zeros(z, 2 * n_z1, name="conv")
    mean, logsd = tf.split(h, 2, axis=-1)

    rescale = tf.get_variable("rescale", [], initializer=tf.constant_initializer(1.))
    scale_shift = tf.get_variable("scale_shift", [], initializer=tf.constant_initializer(0.))
    logsd = tf.tanh(logsd) * rescale + scale_shift

    return gaussian_diag(mean, logsd)


def prior(y_onehot, hps, name=None):
    n_z = hps.top_shape[-1]

    h = tf.zeros([tf.shape(y_onehot)[0]]+hps.top_shape[:2]+[2*n_z])
    h = conv2d_zeros(h, 2*n_z, name="p")
    if hps.ycond:
        h += tf.reshape(linear_zeros(y_onehot, 2*n_z, name="y_emb"), [-1, 1, 1, 2 * n_z])
    mean, logsd = tf.split(h, 2, axis=-1)

    rescale = tf.get_variable("rescale", [], initializer=tf.constant_initializer(1.))
    scale_shift = tf.get_variable("scale_shift", [], initializer=tf.constant_initializer(0.))
    logsd = tf.tanh(logsd) * rescale + scale_shift

    pz = gaussian_diag(mean, logsd)
    logp = lambda z1: pz.logp(z1)
    eps = lambda z1: pz.get_eps(z1)
    sample = lambda eps: pz.sample(eps)

    return logp, sample, eps
