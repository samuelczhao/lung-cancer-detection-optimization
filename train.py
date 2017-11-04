#!/usr/bin/env python3
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np

from dataloader_cropped_mass import async_data_loader, convert_lbl_to_ind
from dataloader_cropped_mass import load_images_with_labels, clean_data
from model import SmallConvNet, ImageClassifier, DenseNet
import time
import matplotlib.pyplot as plt
import scipy.misc


def analyze_image_data():
    '''
        Plots some useful statistics about the images...
    '''
    Xdat, Ydat = load_images_with_labels(0, 40, verbose=True)
    Ydat = np.array(convert_lbl_to_ind(Ydat))
    pdat = Xdat
    Xdat = list(map(lambda i: 256.0 * (i - i.min()) / (i.max() - i.min()), Xdat))
    Xdat = np.array(list(map(
        lambda i: scipy.misc.imresize(i, (226, 226)), Xdat)))

    for i in range(60):
        plt.subplot(133)
        plt.imshow(pdat[i])

        plt.subplot(131)
        plt.hist(Xdat.flatten(), bins=300)
        plt.subplot(132)
        plt.imshow(Xdat[i], cmap='Greys')
        plt.show()


def evaluate_cv_stats(sess, start, end, model, batch_size=32):
    total_accuracy = 0.0
    total_error = 0.0
    cvX, cvY = load_images_with_labels(start, end, verbose=True)
    cvX, cvY = clean_data(cvX, cvY)
    for i in range(0, cvX.shape[0], batch_size):
        acc, err = sess.run([model.accuracy, model.error],
                feed_dict={model.X: cvX[i:i+batch_size],
                           model.Y: cvY[i:i+batch_size],
                           model.m.training: False})
        total_accuracy += min(batch_size, cvX.shape[0] - i) * acc
        total_error += min(batch_size, cvX.shape[0] - i) * err
    return total_accuracy / cvX.shape[0], total_error / cvX.shape[0]


def train(M, out_classes=3, cv_iters=100):
    m = ImageClassifier(M, out_classes)
    global_step = m.global_step
    train_op = tf.train.MomentumOptimizer(1e-2, 0.9).minimize(m.error,
            global_step=global_step)

    summary_op = tf.summary.merge_all()
    logdir = os.path.join("tflogs", m.name)
    sw = tf.summary.FileWriter(logdir)
    sv = tf.train.Supervisor(summary_op=None,
            summary_writer=None,
            logdir=logdir,
            global_step=global_step,
            save_model_secs=120)

    with sv.managed_session() as sess:
        data_gen = async_data_loader(size=(226, 226), start=0, end=1000, batch_size=32)
        sw.add_graph(sess.graph)
        while not sv.should_stop():
            start_time = time.perf_counter()
            imgs, lbls = next(data_gen)
            print("\rData loading time: {}".format(
                time.perf_counter() - start_time), end="")
            lbls = np.array(convert_lbl_to_ind(lbls))
            s, _ = sess.run([summary_op, train_op], feed_dict={m.X: imgs, m.Y: lbls})
            sw.add_summary(s, global_step=sess.run(global_step))
            if sess.run(global_step) % cv_iters == 1:
                print()
                cvAcc, cvErr = evaluate_cv_stats(sess, 1000, 1318, m)
                cv_sum = tf.Summary(value=[
                    tf.Summary.Value(tag="cv_accuracy", simple_value=cvAcc),
                    tf.Summary.Value(tag="cv_error", simple_value=cvErr)])
                sw.add_summary(cv_sum, global_step=sess.run(global_step))
        print()


if __name__ == '__main__':
    train(DenseNet)
