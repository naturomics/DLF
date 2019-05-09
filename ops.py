import numpy as np
import scipy.linalg
import tensorflow as tf
try:
    import horovod.tensorflow as hvd
except ImportError: 
    pass
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope


def shape(inputs, name=None):
    with tf.name_scope(name, "shape"):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)


def allreduce_sum(x):
    if hvd.size() == 1:
        return x
    return hvd.mpi_ops._allreduce(x)


def allreduce_mean(x):
    return allreduce_sum(x) / hvd.size()


def squeeze2d(inputs, factor=2):
    assert factor >= 1
    if factor == 1:
        return inputs
    shape = inputs.get_shape()
    height, width, channels = int(shape[1]), int(shape[2]), int(shape[3])
    assert height % factor == 0 and width % factor == 0
    inputs = tf.reshape(inputs, [-1, height//factor, factor, width//factor, factor, channels])
    inputs = tf.transpose(inputs, [0, 1, 3, 5, 2, 4])
    inputs = tf.reshape(inputs, [-1, height//factor, width//factor, channels*factor*factor])
    return inputs


def unsqueeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height, width, channels = int(shape[1]), int(shape[2]), int(shape[3])
    assert channels >= 4 and channels % 4 == 0
    x = tf.reshape(x, (-1, height, width, int(channels/factor**2), factor, factor))
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, (-1, int(height*factor), int(width*factor), int(channels/factor**2)))
    return x


def default_initializer(std=0.05):
    return tf.random_normal_initializer(0., std)


@add_arg_scope
def linear(name, x, width, do_weightnorm=True, do_actnorm=True, initializer=None, use_bias=True, scale=1.):
    initializer = initializer or default_initializer()
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[1])
        w = tf.get_variable("W", [n_in, width],
                            tf.float32, initializer=initializer)
        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0])
        x = tf.matmul(x, w)
        if use_bias:
            x += tf.get_variable("b", [1, width],
                                 initializer=tf.zeros_initializer())
        if do_actnorm:
            x = actnorm(x, scale=scale, name="actnorm")
        return x


@add_arg_scope
def linear_zeros(x, width, logscale_factor=3, name=None):
    with tf.variable_scope(name, "linear_zeros"):
        n_in = int(x.get_shape()[1])
        w = tf.get_variable("W", [n_in, width], tf.float32,
                            initializer=tf.zeros_initializer())
        x = tf.matmul(x, w)
        x += tf.get_variable("b", [1, width],
                             initializer=tf.zeros_initializer())
        x *= tf.exp(tf.get_variable("logs",
                                    [1, width], initializer=tf.zeros_initializer()) * logscale_factor)
        return x


def add_edge_padding(x, filter_size):
    assert filter_size[0] % 2 == 1
    if filter_size[0] == 1 and filter_size[1] == 1:
        return x
    a = (filter_size[0] - 1) // 2  # vertical padding size
    b = (filter_size[1] - 1) // 2  # horizontal padding size
    in_shape = x.get_shape().as_list()
    x = tf.pad(x, [[0, 0], [a, a], [b, b], [0, 0]])
    name = "_".join([str(dim) for dim in [a, b, *in_shape[1:3]]])
    pads = tf.get_collection(name)
    if not pads:
        pad = np.zeros([1] + x.get_shape().as_list()[1:3] + [1], dtype='float32')
        pad[:, :a, :, 0] = 1.
        pad[:, -a:, :, 0] = 1.
        pad[:, :, :b, 0] = 1.
        pad[:, :, -b:, 0] = 1.
        pad = tf.convert_to_tensor(pad)
        tf.add_to_collection(name, pad)
    else:
        pad = pads[0]
    pad = tf.tile(pad, [tf.shape(x)[0], 1, 1, 1])
    x = tf.concat([x, pad], axis=3)

    return x


@add_arg_scope
def conv2d(x, width,
           filter_size=[3, 3],
           stride=[1, 1],
           pad="SAME",
           do_weightnorm=False,
           do_actnorm=True,
           context1d=None,
           skip=1,
           edge_bias=True,
           name=None):
    with tf.variable_scope(name, "conv2d"):
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[3])

        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable("W", filter_shape, tf.float32,
                            initializer=default_initializer())
        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0, 1, 2])
        if skip == 1:
            x = tf.nn.conv2d(x, w, stride_shape, pad, data_format='NHWC')
        else:
            assert stride[0] == 1 and stride[1] == 1
            x = tf.nn.atrous_conv2d(x, w, skip, pad)
        if do_actnorm:
            x = actnorm(x, name="actnorm")
        else:
            x += tf.get_variable("b", [1, 1, 1, width],
                                 initializer=tf.zeros_initializer())

        if context1d != None:
            context = tf.reshape(linear("context", context1d,
                                        width), [-1, 1, 1, width])
            x += context
    return x


@add_arg_scope
def conv2d_zeros(x,
                 width,
                 filter_size=[3, 3],
                 stride=[1, 1],
                 pad="SAME",
                 logscale_factor=3,
                 skip=1,
                 edge_bias=True,
                 name=None):
    with tf.variable_scope(name, "conv2d"):
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[3])
        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable("W", filter_shape, tf.float32,
                            initializer=tf.zeros_initializer())
        if skip == 1:
            x = tf.nn.conv2d(x, w, stride_shape, pad, data_format='NHWC')
        else:
            assert stride[0] == 1 and stride[1] == 1
            x = tf.nn.atrous_conv2d(x, w, skip, pad)
        x += tf.get_variable("b", [1, 1, 1, width],
                             initializer=tf.ones_initializer())
        x *= tf.exp(tf.get_variable("logs",
                                    [1, width], initializer=tf.zeros_initializer()) * logscale_factor)
    return x


@add_arg_scope
def actnorm(x,
            scale=1.,
            logdet=None,
            logscale_factor=3.,
            batch_variance=False,
            reverse=False,
            context1d=None,
            trainable=True,
            name=None):
    with tf.variable_scope(name, "actnorm"):
        shape = x.get_shape()
        rank = len(shape)
        assert rank == 2 or rank == 4
        _shape = [1 for i in range(rank - 1)] + [shape[-1]]
        logdet_factor = 1 if rank == 2 else int(shape[1])*int(shape[2])
        axis = [0] if rank == 2 else [0,1,2]

        version = 1
        if version == 0:
            b = tf.get_variable("b", _shape, initializer=tf.constant_initializer(0.))
            logs = tf.get_variable("logs", _shape) #, initializer=tf.constant_initializer(0.))

            if context1d is not None:
                logs = logs + tf.reshape(linear("context_s", context1d, _shape[-1], do_actnorm=False, use_bias=False), [-1] + _shape[1:])
                b = b + tf.reshape(linear("context_b", context1d, _shape[-1], do_actnorm=False, use_bias=False), [-1] + _shape[1:])
            b = tf.nn.tanh(b)

            logs = -tf.nn.sigmoid(logs)
        elif version == 1:
            b = tf.get_variable("b", _shape)
            logs = tf.get_variable("logs", _shape, initializer=tf.constant_initializer(-1.)) * logscale_factor


        if not reverse:
            x += b
            x = x * tf.exp(logs)
        else:
            x = x * tf.exp(-logs)
            x -= b

        if logdet is not None:
            dlogdet = tf.reduce_sum(logs) * logdet_factor
            if reverse:
                dlogdet *= -1
            # if not reverse and False:
            #     tf.summary.scalar("dlogdet", dlogdet)
            logdet += dlogdet
            return x, logdet
        else:
            return x


@add_arg_scope
def invertible_1x1_conv(z, logdet, reverse=False, name=None, use_bias=False):
    with tf.variable_scope(name, "invconv"):
        shape = z.get_shape().as_list()
        w_shape = [shape[3], shape[3]]

        # Sample a random orthogonal matrix:
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
        w = tf.get_variable("W", dtype=tf.float32, initializer=w_init)

        det_w = tf.matrix_determinant(tf.cast(w, 'float64'))
        dlogdet = tf.cast(tf.log(abs(det_w)), 'float32') * shape[1] * shape[2]

        if use_bias:
            b = tf.get_variable("bias", [1, 1, 1, shape[3]])

        if not reverse:
            _w = w[tf.newaxis, tf.newaxis, ...]
            z = tf.nn.conv2d(z, _w, [1, 1, 1, 1], 'SAME', data_format='NHWC')
            logdet += dlogdet
            if use_bias:
                z += b
        else:
            if use_bias:
                z -= b
            w_inv = tf.matrix_inverse(w)
            _w = w_inv[tf.newaxis, tf.newaxis, ...]
            z = tf.nn.conv2d(z, _w, [1, 1, 1, 1], 'SAME', data_format='NHWC')
            logdet -= dlogdet
        return z, logdet


def flatten_sum(logps):
    if len(logps.get_shape()) == 2:
        return tf.reduce_sum(logps, [1])
    elif len(logps.get_shape()) == 4:
        return tf.reduce_sum(logps, [1, 2, 3])
    else:
        raise Exception()


def gaussian_diag(mean, logsd):
    class o(object):
        pass
    o.mean = mean
    o.logsd = logsd
    o.eps = tf.random_normal(tf.shape(mean))
    def sample(eps=None):
        epsilon = eps if eps is not None else o.eps
        return mean + tf.exp(logsd) * epsilon

    o.sample = sample
    o.logps = lambda x: -0.5*(np.log(2 * np.pi) + 2. * logsd + tf.square(x - mean) * tf.exp(-2.*logsd))
    o.logp = lambda x: flatten_sum(o.logps(x))
    o.get_eps = lambda x: (x - mean) * tf.exp(-logsd)
    return o
