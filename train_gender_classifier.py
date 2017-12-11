# -*- coding: utf-8 -*-

import argparse
import sys

import tensorflow as tf

from model import utils
from tensorflow.contrib import slim

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

    with tf.Graph().as_default() as graph:
        utils.load_model(model=args.model_path)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name="gender_label")

        logits = gender_model(embeddings)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits,
                                                                       name="softmax_cross_entropy")
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_losses = tf.add_n([cross_entropy_mean] + regularization_losses)

        session = tf.Session(graph=graph)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        update_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "gender_model")

        session.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help="The model of calculating the embedding vector")
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help="Indicates if a new classifier should be trained or a classification", default='CLASSIFY')
    parser.add_argument('--train_data_dir', type=str, help='train data directory', default="~/data/imdb_corp")

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
