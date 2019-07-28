from importlib import import_module
import tensorflow as tf
import argparse
import numpy as np
from time import time
import sys
import os

from model import Model
import graphics

class ResultLogger(object):
    def __init__(self, path):
        self.log_fd = open(path, 'w')
        self.log_fd.write("epoch,step,loss,bits_x,bits_y,l2_loss\n")

    def write(self, epoch, results, step=None):
        string = "{:d}".format(epoch)
        if step is not None:
            string +=",{:d}".format(step)
        for i in range(len(results)):
            string += ",{:.4f}".format(results[i])
        string += "\n"
        self.log_fd.write(string)
        self.log_fd.flush()

    def close(self):
        self.log_fd.close()


# print status to screen
def _print(results, epoch, step, speed):
    log_str = "\x1b[1A\x1b[2K "
    log_str += "epoch, step, loss, bits_x, bits_y, l2_loss, speed(samples/sec)\n"
    log_str += " {:3d}, {:6d}".format(epoch, step)
    for i in range(len(results)):
        log_str += ", {:.3f}".format(results[i])
    log_str += ", {:.3f}\r".format(speed)
    sys.stdout.write(log_str)
    sys.stdout.flush()


def init_visualizations(sess, model, batch_size, path, hps=None):
    """
    Randomly sampling during training
    """
    rows = 10
    cols = rows
    total_batch = rows*cols

    epsilon = tf.placeholder(tf.float32, [None]+hps.top_shape, name="prior/epsilon")
    labels = tf.placeholder(tf.int32, [None], name="prior/label")
    if hps.problem == "celeba" and hps.conditioning:
        attr = np.random.choice([-1., 1.], size=[total_batch, 40])
        condition = tf.placeholder(tf.float32, [None, 40], name="prior/condition")
    else:
        condition = None
    model.decode(labels=labels, condition=condition,
                 epsilon=[None]*(hps.num_levels-1)+[epsilon])
    os.makedirs(path, exist_ok=True)

    y = np.asarray([_y % model.num_classes for _y in (list(range(cols)) * rows)], dtype='int32')
    eps = np.random.normal(size=[total_batch] + hps.top_shape)
    temperatures = [0., .25, .35, .4, .5, .6, .7, .8, .9, 1.]

    def sample_batch(eps):
        xs = []
        for i in range(int(np.ceil(total_batch / batch_size))):
            start = i * batch_size
            end = (i + 1) * batch_size
            end = end if end <= total_batch else total_batch
            if hps.problem == "celeba" and hps.conditioning:
                gen_x = sess.run(model.gen_x, feed_dict={labels:y[start:end], epsilon:eps[start:end],
                                                         condition:attr[start:end]})
            else:
                gen_x = sess.run(model.gen_x, feed_dict={labels:y[start:end], epsilon:eps[start:end]})
            xs.append(gen_x)
        return np.concatenate(xs, axis=0)

    def draw_samples(epoch):
        x_samples = []
        # eps = np.random.normal(size=[total_batch] + hps.top_shape)
        for i, t in enumerate(temperatures):
            x_sample = sample_batch(t * eps)
            x_sample = np.reshape(x_sample, (total_batch, model.height, model.width, model.channels))
            fname = 'epoch_{}_sample_{}.png'.format(epoch, i)
            graphics.save_raster(x_sample, os.path.join(path, fname))
            x_samples.append(x_sample)
        return np.concatenate(x_samples, axis=0)
    return draw_samples


def train(model, dataloader, sess, hps):
    train_iterator = dataloader(hps.batch_size * hps.num_gpus, mode="train")
    test_iterator = dataloader(hps.batch_size * hps.num_gpus, mode="eval")
    images, labels = dataloader.get_element("images", "labels")
    if hps.problem == 'celeba' and hps.conditioning:
        hps.ycond = False
        model.train(images, labels, condition=dataloader.get_element("attr"))
    else:
        model.train(images, labels)
    print("\nNumber of trainable parameters: %s" % model.num_variables)

    train_handle = dataloader.initialize(sess, train_iterator, get_handle=True)
    test_handle = dataloader.initialize(sess, test_iterator, get_handle=True)

    # Sampling during training (one time per epoch as default), optional.
    # It can take some memory on GPU:0. If you have GPUs with small memory,
    # it's recommended to comment out all the related 'visualize' code
    # with tf.device("/cpu:0"):
    visualize = init_visualizations(sess=sess, model=model,
                                    batch_size=hps.batch_size, hps=hps,
                                    path=os.path.join(hps.results_dir, "samples"))

    log_path = os.path.join(hps.results_dir, 'train_logs')
    global_stats = [model.total_loss, model.bits_x, model.bits_y, model.l2_loss]
    train_logger = ResultLogger(os.path.join(hps.results_dir, 'train.csv'))
    test_logger = ResultLogger(os.path.join(hps.results_dir, 'test.csv'))

    loss = loss_best = test_loss_best = 999999.
    epoch, step, msg = model.initialize(sess, log_path)
    print(msg)
    results = []
    start_time = time()
    lr_up1 = lr_up2 = 1.0
    while epoch <= hps.num_epoch:
        ## learning rate scheduler
        # if `steps_warmup` is non-zero, we use a two-stage warmup strategy to
        # automatically search an appropriate upper bound of exponential decay
        # learning rate, or use the given one by `--lr` argument
        if step <= hps.steps_warmup:
            if loss < loss_best and step < hps.steps_warmup:
                loss_best = loss
                lr_up1 = lr
            elif step == hps.steps_warmup:
                loss_best = 999999.
            lr = lr * ((1.0/1e-5)**(1./hps.steps_warmup)) if step > 1 else 1e-5
            lr = min(lr, hps.initial_lr) if step != hps.steps_warmup else 1e-5
        elif step <= 2*hps.steps_warmup:
            if loss < loss_best and step < 2*hps.steps_warmup:
                loss_best = loss
                lr_up2 = lr
            elif step == 2*hps.steps_warmup:
                hps.initial_lr = 0.5*(lr_up1 + lr_up2)
            lr = lr * ((1.0/1e-5)**(1./hps.steps_warmup))
            lr = min(lr, lr_up1)
        else:
            # exponential decay learning rate
            lr = max(hps.initial_lr * np.power(hps.decay_rate, step/hps.decay_steps), hps.lr)

        try:
            _, *stats = sess.run([model.train_ops] + global_stats,
                                 feed_dict={dataloader.handle:train_handle, model.lr:lr})
            results.append(stats)

            if hps.print_per_steps > 0 and step % hps.print_per_steps == 0:
                duration = time() - start_time
                speed = hps.batch_size * hps.num_gpus * hps.print_per_steps / duration
                _print(results[-1], epoch, step, speed)
                start_time = time()
            if hps.problem in ["celeba", "imagenet32x32", "imagenet64x64"] and step % hps.steps_train_sum == 0:
                visualize(epoch)
                model.save(sess, log_path, epoch, step)

            if step % hps.valid_per_steps == 0 and hps.valid_per_steps > 0:
                dataloader.initialize(sess, test_iterator)
                test_results = []
                while True:
                    try:
                        stats = sess.run(global_stats, feed_dict={dataloader.handle:test_handle})
                        test_results.append(stats)
                    except tf.errors.OutOfRangeError:
                        break
                test_results = np.nanmean(test_results, axis=0)
                test_logger.write(epoch, test_results, step=step)

        except tf.errors.OutOfRangeError:
            results = np.nanmean(results, axis=0)
            train_logger.write(epoch, results, step=step)
            visualize(epoch)

            if epoch % hps.test_per_epochs == 0 and hps.test_per_epochs > 0:
                dataloader.initialize(sess, test_iterator)
                test_results = []
                while True:
                    try:
                        stats = sess.run(global_stats, feed_dict={dataloader.handle:test_handle})
                        test_results.append(stats)
                    except tf.errors.OutOfRangeError:
                        break
                test_results = np.nanmean(test_results, axis=0)
                test_logger.write(epoch, test_results, step=step)
                if test_results[0] < test_loss_best:
                    model.test_saver.save(sess, os.path.join(hps.results_dir, "logs", "model_best_loss"))
                    test_loss_best = test_results[0]
            model.save(sess, log_path, epoch, step)
            dataloader.initialize(sess, train_iterator)
            results = []
            epoch += 1
        except tf.errors.InvalidArgumentError as e:
            print(e)
            exit()

        step += 1
    train_logger.close()
    test_logger.close()


def infer(model, dataloader, sess, hps):
    iterator = dataloader(hps.batch_size * hps.num_gpus, mode="test")
    x, labels = dataloader.get_element("images", "labels")
    if hps.problem == "celeba" and hps.conditioning:
        condition = dataloader.get_element("attr")
    else:
        condition = None
    z = model.encode(x, labels, condition=condition)

    model.initialize(sess, os.path.join(hps.results_dir, 'logs'))
    handle = dataloader.initialize(sess, iterator, get_handle=True)
    zs = []
    while True:
        try:
            z = sess.run(z, feed_dict={dataloader.handle:handle})
            zs.append(z)
        except tf.errors.OutOfRangeError:
            break
    z = np.concatenate(zs, axis=0)
    np.save(os.path.join(hps.results_dir, "latent.npy"), z)
    return zs


def main(hps):
    tf.set_random_seed(hps.seed + hps.num_gpus * hps.batch_size)
    np.random.seed(hps.seed + hps.num_gpus * hps.batch_size)
    if hps.problem == "imagenet32x32" or hps.problem == "imagenet64x64":
        dataloader = import_module("datasets.imagenet").DataLoader(path=hps.data_dir,
                                                                   threads_fmap=hps.threads_fmap,
                                                                   threads_dmap=hps.threads_dmap,
                                                                   buffer_size=hps.buffer_size,
                                                                   image_size=hps.problem.split('x')[-1])
    else:
        dataloader = import_module("datasets." + hps.problem).DataLoader(path=hps.data_dir,
                                                                         threads_fmap=hps.threads_fmap,
                                                                         threads_dmap=hps.threads_dmap,
                                                                         buffer_size=hps.buffer_size)
    hps.num_classes = dataloader.num_classes
    model = Model(hps)

    # Create tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if hps.debug:
        from tensorflow.python import debug as tf_debug
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    if not hps.inference:
        train(model, dataloader, sess, hps)
    else:
        infer(model, dataloader, sess, hps)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inference", action='store_true',
                        help="Switch to inference mode")
    parser.add_argument("--debug", action='store_true',
                        help="Set tf.Session() in debug mode")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Path to store results, the root directory for all training and inference outputs")
    parser.add_argument("--print_per_steps", type=int, default=100,
                        help="Print training status to screen per steps. Set to <=0 int to switch off")
    parser.add_argument("--test_per_epochs", type=int, default=1,
                        help="Test model per epochs")
    parser.add_argument("--valid_per_steps", type=int, default=3000,
                        help="Evaluation per steps")
    parser.add_argument("--steps_train_sum", type=int, default=10000,
                        help="Summary training per steps")
    parser.add_argument("--buffer_size", type=int, default=50000,
                        help="Buffer size of tf.data.dataset shuffle")
    parser.add_argument("--seed", type=int, default=199512, help="Random seed")

    # Dataset and dataloader
    parser.add_argument("--problem", type=str, default='cifar10',
                        help="Dataset to use (mnist/cifar10/celeba/imagenet32x32/imagenet64x64/lsun)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory of tfrecord data")
    parser.add_argument("--threads_fmap", type=int, default=2,
                        help="Number of threads for parallel file reading")
    parser.add_argument("--threads_dmap", type=int, default=4,
                        help="number of threads for parallel dataset map")

    # Optimization hyperparams
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Minibatch size per GPU, total batch size=batch_size * num_gpus")
    parser.add_argument("--num_epoch", type=int, default=10000,
                        help="Number of training epochs")
    parser.add_argument("--steps_warmup", type=int, default=0,
                        help="Warmup steps")
    parser.add_argument("--initial_lr", type=float, default=0.05,
                        help="Initial learning rate")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Base learning rate")
    parser.add_argument("--decay_steps", type=int, default=1000,
                        help="Exponential decay steps")
    parser.add_argument("--decay_rate", type=float, default=0.93,
                        help="Exponential decay rate")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs")

    parser.add_argument("--optimizer", type=str, default="adamax",
                        help="adam or adamax")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--weight_decay", type=float, default=1.0,
                        help="Weight decay. Switched off by default")

    parser.add_argument("--l2_factor", type=float, default=0.,
                        help="Factor for L2 regularization, beta=l2_factor/number of variables included in L2 regularization")

    # Model hyperparams
    parser.add_argument("--num_bits_x", type=int, default=8,
                        help="Number of bits of x")

    parser.add_argument("--depth", type=int, default=32,
                        help="Depth of network of per flow")
    parser.add_argument("--num_levels", type=int, default=3,
                        help="Number of levels")
    parser.add_argument("--affine_coupling", action="store_true",
                        help="Let h be identity when K=2, where dynamic linear transformation turns out to be affine coupling layer.")

    parser.add_argument("--invconv_bias", action='store_true',
                        help="Use bias for invertiable 1x1 convolutions")

    parser.add_argument("--num_parts", type=int, default=2,
                        help="Number of parts: split x into K (2/4/6/8) parts")
    parser.add_argument("--splitting", type=int, default=0,
                        help="Fraction of dimension of each sub-part(only works when num_parts=4):\
                        0=D/num_parts for each x_k; 1=incremental sequence, ie. D/8, D/8, D/4, D/2;\
                        2=decrement sequence, ie. D/2, D/4, D/8, D/8")
    parser.add_argument("--width", type=int, default=512,
                        help="Width/Channels of hidden layers in NN() of flow step")

    parser.add_argument("--conditioning", action="store_true",
                        help="Conditional dynamic linear transformation")
    parser.add_argument("--decomposition", type=int, default=0,
                        help="Dynamic linear transformation type: 0=non-inverse, 1=inverse")

    parser.add_argument("--ycond", action='store_true',
                        help="log p(y|x)")
    parser.add_argument("--weight_y", type=float, default=0.01,
                        help="Weight of log p(y|x) in weighted loss")

    return parser.parse_args()


if __name__ == "__main__":
    main(hps=get_arguments())
