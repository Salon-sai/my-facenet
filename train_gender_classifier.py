# -*- coding: utf-8 -*-

import argparse
import sys
import os

import tensorflow as tf
import scipy.io as sio

import optimizer

from model import utils
from tensorflow.contrib import slim
import datetime

def load_data(mat_path):
    mat_data = sio.loadmat(mat_path)
    imdb_data = mat_data['imdb']
    image_paths = imdb_data[0][0][2]
    gender_data = imdb_data[0][0][3]
    print(gender_data.shape)

def gender_model(embeddings):
    with tf.variable_scope("gender_model"):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.random_normal_initializer(),
                            biases_initializer=tf.constant_initializer(),
                            weights_regularizer=slim.l2_regularizer(scale=0.5)):
            net = slim.fully_connected(embeddings, num_outputs=1024, scope="hidden")
            net = slim.fully_connected(net, num_outputs=2, activation_fn=None, scope="logits")
    return net

def main(args):
    imdb_dir = os.path.expanduser(args.imdb_corp_root)
    mat_path = os.path.join(imdb_dir, 'imdb.mat')
    load_data(mat_path)
    exit()

    subdir = datetime.datetime.strftime(datetime.datetime.now(), 'gender-%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    with tf.Graph().as_default() as graph:
        utils.load_model(model=args.model_path)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        global_step = tf.Variable(0, trainable=False)
        labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name="gender_label")

        logits = gender_model(embeddings)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits,
                                                                       name="softmax_cross_entropy")
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_losses = tf.add_n([cross_entropy_mean] + regularization_losses)
        update_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "gender_model")

        train_op = optimizer.train(total_loss=total_losses,
                                   global_step=global_step,
                                   optimizer=args.optimizer,
                                   learning_rate=0.5,
                                   moving_average_decay=0.99,
                                   update_gradient_vars=update_vars)

        saver = tf.train.Saver(update_vars, max_to_keep=3)

        summary_op = tf.summary.merge_all()

        session = tf.Session(graph=graph)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        summary_writer = tf.summary.FileWriter(logdir=log_dir, graph=session.graph)

        session.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help="The model of calculating the embedding vector")
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help="Indicates if a new classifier should be trained or a classification", default='CLASSIFY')
    parser.add_argument('--imdb_corp_root', type=str, help='The directory of imdb crop data', default="~/data/imdb_crop")
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument("--logs_base_dir", type=str, help='Directory where to write event logs.', default='logs/')


    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
