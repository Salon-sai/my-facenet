# -*- coding: utf-8 -*-

import sys
import os
import argparse

import tensorflow as tf
import numpy as np

import metrics
import matplotlib.pyplot as plt
import sklearn

from scipy import misc
from model import utils

def main(args):
    with tf.Graph().as_default():

        with tf.Session() as session:

            data_dir = os.path.expanduser(args.data_dir)
            paths, actual_issame = load_pair(data_dir)
            actual_issame = np.asarray(actual_issame)
            utils.load_model(os.path.expanduser(args.model))

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]

            print("Running forward pass on XiaoYu images")
            batch_size = args.batch_size
            nrof_images = len(paths)
            nrof_batch = int(np.ceil(nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batch):
                start = i * batch_size
                end = min((i + 1) * batch_size, nrof_images)
                images_path = paths[start: end]
                images = np.zeros((end - start, image_size, image_size, 3))
                for j, image_path in enumerate(images_path):
                    images[j, :, :, :] = misc.imread(image_path)
                emb_array[start:end, :] = session.run(embeddings, feed_dict={images_placeholder:images, phase_train_placeholder:False})

            thresholds = np.arange(0, 1, 0.01)
            tprs, fprs, accuracy, _, _, _, _ = metrics.evaluate(emb_array, actual_issame)

            plt.subplot(211)
            auc = sklearn.metrics.auc(fprs, tprs)
            best_threshold_index = np.argmax(accuracy)
            print("When threshold %1.3f, we get best accuracy: %1.3f, true positive rate %1.3f, false positive rate %1.3f"
                  % (thresholds[best_threshold_index], np.max(accuracy), tprs[best_threshold_index], fprs[best_threshold_index]))
            print("Number of positive sample : %d, Number of positive sample : %d"
                  % (int(np.sum(actual_issame)), int(np.sum(np.logical_not(actual_issame)))))

            plt.plot(fprs, tprs, color="r", label="ROC")
            plt.plot([0, 1], [0, 1], '--',color="b", label="Base")
            plt.ylabel("True Positive")
            plt.xlabel("False Positive")
            plt.legend()
            plt.title('Area Under Curve (AUC): %1.3f' % auc)

            plt.subplot(234)
            plt.plot(thresholds, accuracy, label="Accuracy")
            plt.xlabel("Threshold")
            plt.title("Mean Accuracy with Standard Deviation: %1.3f+-%1.3f" % (np.mean(accuracy), np.std(accuracy)))

            plt.subplot(235)
            plt.plot(thresholds, tprs, label="True Positive Rate (TPR)")
            plt.xlabel("Threshold")
            plt.title("Mean TPR with Standard Deviation: %1.3f+-%1.3f" % (np.mean(tprs), np.std(tprs)))

            plt.subplot(236)
            plt.plot(thresholds, tprs, label="False Positive Rate (FPR)")
            plt.xlabel("Threshold")
            plt.title("Mean FPR with Standard Deviation: %1.3f+-%1.3f" % (np.mean(fprs), np.std(fprs)))

            plt.show()

def load_image(image_paths, image_size):
    images = np.zeros((len(image_paths), image_size, image_size, 3))
    for i, image_path in enumerate(image_paths):
        image = misc.imread(image_path)
        if image.shape[1] > image_size:
            sz1 = int(image.shape[1] // 2)
            sz2 = int(image_size // 2)
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
            image = image[(sz1 - sz2 + v): (sz1 + sz2 + v), (sz1 - sz2 + h) : (sz1 + sz2 + h), :]
            image = prewhiten(image)
        images[i, :, :, :] = image

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def load_pair(data_dir):
    actual_issame = []
    pairs = []
    issame = False
    for pos_or_neg in os.listdir(data_dir):
        if pos_or_neg == "true":
            issame = True
        else:
            issame = False
        label_dir = os.path.join(data_dir, pos_or_neg)  # the dir of "true" or "false"
        num_pair = 0
        for sample_id in os.listdir(label_dir):
            sample_dir = os.path.join(label_dir, sample_id)
            if not os.path.isdir(sample_dir):
                continue
            images_paths = [os.path.join(sample_dir, image_path) for image_path in os.listdir(sample_dir)]
            if len(images_paths) < 2:
                print("The sample dir %s has not enough images" % sample_dir)
                continue
            pairs += (images_paths[0], images_paths[1])
            num_pair += 1
        actual_issame += [issame] * num_pair
    assert len(actual_issame) * 2 == len(pairs)
    return pairs, actual_issame


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned Xiaoyu face patches.')
    parser.add_argument('model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch in the XiaoYu test set.', default=20)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

