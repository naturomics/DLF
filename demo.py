import tensorflow as tf
from time import time
import numpy as np
import argparse
import os
from glob import glob
import pandas as pd
from PIL import Image

from model import Model
from main import get_arguments


def get_batch(files):
    imgs = []
    for f in files:
        imgs.append(np.reshape(np.array(Image.open(f)), [256, 256, 3]))
    return np.array(imgs)


def main(hps):
    # Setup tf session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    x = tf.placeholder(tf.uint8, [None,256,256,3], name="input/x")
    label = tf.placeholder(tf.int32, [None], name="input/label")
    model = Model(hps)
    eps = model.encode(x, label)
    epsilon = [tf.placeholder(tf.float32, z.get_shape().as_list(), name="input/eps_%s"%i) for i, z in enumerate(eps)]
    gen_x = model.decode(label, epsilon=epsilon)

    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join(hps.results_dir, "train_logs")))

    pair, files = 1, ["demo/raw_imgs/01.png", "demo/raw_imgs/02.png"]
    # pair, files = 2, ["demo/raw_imgs/04.png", "demo/raw_imgs/05.png"]
    # pair, files = 3, ["demo/raw_imgs/06.png", "demo/raw_imgs/07.png"]
    # pair, files = 4, ["demo/raw_imgs/06.png", "demo/raw_imgs/03.png"]
    for i in range(int(np.ceil(len(files)/hps.batch_size))):
        start = i * hps.batch_size
        end = (i + 1) * hps.batch_size
        end = end if end <= len(files) else len(files)
        imgs = get_batch(files[start:end])
        filenames = [os.path.basename(f).replace('.png', '') for f in files[start:end]]
        y = np.zeros(imgs.shape[0])

        z = sess.run(eps, feed_dict={x:imgs, label:y})

        y = y[:1]
        feed_dict = {label:y}
        manipulator = []
        for i, dec_eps in enumerate(epsilon):
            manipulator.append(z[i][0] - z[i][1])

        times = []
        scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(1,50):
            for j, dec_eps in enumerate(epsilon):
                feed_dict[dec_eps] = np.expand_dims(z[j][1] + i/50*manipulator[j], axis=0)
            start_time = time()
            gen_imgs = sess.run(gen_x, feed_dict)
            times.append(time()-start_time)
            img = Image.fromarray(gen_imgs[0])
            img.save("demo/pairs%s/interpolation_%s.png"%(pair, i))

            latent = [latent[0] + np.random.normal(scale=scales[i//8], size=latent[0].shape) for latent in z]
            latent[-1] = z[-1][0]
            for j, dec_eps in enumerate(epsilon):
                feed_dict[dec_eps] = np.expand_dims(latent[j], axis=0)
            gen_imgs = sess.run(gen_x, feed_dict)
            img = Image.fromarray(gen_imgs[0])
            img.save("demo/%s/perturbations_scale_%s_sample_%s.png"%(filenames[0], i//8, i%8))

            latent = [latent[1] + np.random.normal(scale=scales[i//8], size=latent[1].shape) for latent in z]
            latent[-1] = z[-1][1]
            for j, dec_eps in enumerate(epsilon):
                feed_dict[dec_eps] = np.expand_dims(latent[j], axis=0)
            gen_imgs = sess.run(gen_x, feed_dict)
            img = Image.fromarray(gen_imgs[0])
            img.save("demo/%s/perturbations_scale_%s_sample_%s.png"%(filenames[1], i//8, i%8))
            print(i)
        print(np.mean(times[-10:]))


if __name__ == "__main__":
    hps = get_arguments()
    hps.num_classes = 0
    hps.num_levels = 6
    hps.width = 128
    main(hps=hps)
