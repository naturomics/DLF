import tensorflow as tf

from ops import allreduce_mean

# Optimizers

'''
Polyak averaging op
'''

def polyak(params, beta):
    ema = tf.train.ExponentialMovingAverage(decay=beta, zero_debias=True)
    avg_op = tf.group(ema.apply(params))
    # Swapping op
    updates = []
    for i in range(len(params)):
        p = params[i]
        avg = ema.average(p)
        tmp = 0. + avg * 1.
        with tf.control_dependencies([tmp]):
            update1 = avg.assign(p)
            with tf.control_dependencies([update1]):
                update2 = p.assign(tmp)
                updates += [update1, update2]
    swap_op = tf.group(*updates)
    return avg_op, swap_op, ema


def adamax(params, cost_or_grads, alpha=3e-4, hps=None, global_step=None, epsilon=1e-8, allreduce=False):
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads

    beta2 = 1-1./(hps.train_its*hps.polyak_epochs)
    if allreduce:
        grads = [allreduce_mean(g) for g in grads]

    if global_step is not None:
        t = global_step
    else:
        t = tf.Variable(1., trainable=False, name='adam_t')

    alpha_t = alpha * tf.sqrt((1. - tf.pow(beta2, t))) / \
        (1. - tf.pow(hps.beta1, t))
    updates.append(t.assign_add(1))

    for w, g in zip(params, grads):
        if g is None:
            continue
        mom2 = tf.Variable(tf.zeros(w.get_shape()), trainable=False, name=w.name.split(":")[0] + '_adam_m2')
        if hps.beta1 > 0:
            mom1 = tf.Variable(tf.zeros(w.get_shape()), trainable=False, name=w.name.split(":")[0] + '_adam_m1')
            mom1_new = hps.beta1 * mom1 + (1. - hps.beta1) * g
            updates.append(mom1.assign(mom1_new))
        else:
            mom1_new = g
        m2_new = tf.maximum(beta2 * mom2, abs(g))
        delta_t = mom1_new / (m2_new + epsilon)
        w_new = hps.weight_decay * w - alpha_t * delta_t
        updates.append(mom2.assign(m2_new))
        updates.append(w.assign(w_new))

    # Polyak averaging
    polyak_avg_op, polyak_swap_op, ema = polyak(params, beta2)
    train_op = tf.group(polyak_avg_op, *updates)
    return train_op, polyak_swap_op, ema


def adam(params, cost_or_grads, alpha=3e-4, hps=None, global_step=None, epsilon=1e-8, allreduce=False):
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads

    beta2 = 1-1./(hps.train_its*hps.polyak_epochs)
    if allreduce:
        grads = [allreduce_mean(g) for g in grads]

    if global_step is not None:
        t = global_step
    else:
        t = tf.Variable(1., trainable=False, name='adam_t')

    alpha_t = alpha * tf.sqrt((1. - tf.pow(beta2, t))) / (1. - tf.pow(hps.beta1, t))
    updates.append(t.assign_add(1))

    for w, g in zip(params, grads):
        mom2 = tf.Variable(tf.zeros(w.get_shape()), trainable=False, name=w.name + '_adam_m2')
        if hps.beta1 > 0:
            mom1 = tf.Variable(tf.zeros(w.get_shape()), trainable=False, name=w.name + '_adam_m1')
            mom1_new = hps.beta1 * mom1 + (1. - hps.beta1) * g
            updates.append(mom1.assign(mom1_new))
        else:
            mom1_new = g
        m2_new = beta2 * mom2 + (1. - beta2) * tf.square(g)
        delta_t = mom1_new / (tf.sqrt(m2_new) + epsilon)
        w_new = hps.weight_decay * w - alpha_t * delta_t
        updates.append(mom2.assign(m2_new))
        updates.append(w.assign(w_new))

    # Polyak averaging
    polyak_avg_op, polyak_swap_op, ema = polyak(params, beta2)
    train_op = tf.group(polyak_avg_op, *updates)
    return train_op, polyak_swap_op, ema
