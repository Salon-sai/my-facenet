# -*- coding: utf-8 -*-

import argparse
import sys
import os

import tensorflow as tf
import numpy as np

import optimizer

from model import utils
from tensorflow.contrib import slim
import datetime

from data_process.utils import load_data
from data_process.imdb_process import ImageDatabase

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
    imdb_dir = os.path.expanduser(args.imdb_aligned_root)
    image_database = ImageDatabase(imdb_dir, args.max_num_images)
    batch_size = args.batch_size

    subdir = datetime.datetime.strftime(datetime.datetime.now(), 'gender-%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    with tf.Session() as session:
        utils.load_model(model=args.model_path)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        embedding_size = embeddings.get_shape()[1]

        emb_array = np.zeros((image_database.nrof_images, embedding_size))

        nrof_batch = int(np.ceil(image_database.nrof_images / batch_size))

        for i in range(nrof_batch):
            start_index = i * batch_size
            end_index = min(image_database.nrof_images, (i + 1) + batch_size)
            images = load_data(image_database.images_path[start_index: end_index], False, False, args.image_size)
            feed_dict = { images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index: end_index] = session.run(embeddings, feed_dict=feed_dict)

    image_database.embeddings = emb_array
    train_gender(image_database, embedding_size, args.optimizer, args.epoch_size, batch_size, log_dir)


def train_gender(image_database, embedding_size, optimizer_type, max_num_epoch, batch_size, log_dir):
    _, train_embeddings, _, train_genders = image_database.train_data
    _, valid_embeddings, _, valid_genders = image_database.valid_data
    nrof_train_samples = len(train_embeddings)

    print("The number of female: %d" % np.sum(train_genders == 0))
    print("The number of male: %d" % np.sum(train_genders == 1))

    with tf.Graph().as_default() as graph:
        labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name="gender_label")
        embeddings_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, embedding_size],
                                                name="embeddings_placeholder")

        global_step = tf.Variable(0, trainable=False)

        logits = gender_model(embeddings_placeholder)

        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_placeholder))

        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        tf.summary.scalar("accuracy", accuracy)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits,
                                                                       name="softmax_cross_entropy")
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_losses = tf.add_n([cross_entropy_mean] + regularization_losses)
        tf.summary.scalar("loss", total_losses)

        update_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "gender_model")

        train_op = optimizer.train(total_loss=total_losses,
                                   global_step=global_step,
                                   optimizer=optimizer_type,
                                   # optimizer=args.optimizer,
                                   learning_rate=0.5,
                                   moving_average_decay=0.99,
                                   update_gradient_vars=update_vars)

        saver = tf.train.Saver(update_vars, max_to_keep=3)

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(log_dir, graph)

        session = tf.Session(graph=graph)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        epoch = 0
        nrof_batch = int(np.ceil(nrof_train_samples / batch_size))
        while epoch < max_num_epoch:
            for i in range(nrof_batch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_train_samples)
                feed_dict = {
                    labels_placeholder: train_genders[start_index: end_index],
                    embeddings_placeholder: train_embeddings[start_index: end_index],
                }
                batch_loss, _, gs, summary = session.run([total_losses, train_op, global_step, summary_op], feed_dict=feed_dict)

                summary_writer.add_summary(summary, gs)
                print("[%3d/%6d] Batch Loss: %1.4f" % (epoch, gs, batch_loss))

            # evaluate
            acc = session.run(accuracy, feed_dict={
                labels_placeholder: valid_genders,
                embeddings_placeholder: valid_embeddings
            })
            print("----------------\n")
            print("\n")
            print("\t\t Epoch: %3d, Valid Accuracy: %1.4f" % (epoch, acc))
            print("\n")
            print("----------------\n")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help="The model of calculating the embedding vector")
    parser.add_argument('--imdb_aligned_root', type=str, help='The directory of imdb cropped and aligned data',
                        default="~/data/imdb_cropped_clean")
    parser.add_argument("--save_embedding_path", type=str, help="The directory saving embeddings",
                        default="~/data/imdb_cropped_clean_embedd")
    parser.add_argument("--batch_size", type=int, help="The size of train batch", default=100)
    parser.add_argument("--epoch_size", type=int, help="The size of epoch size", default=500)
    parser.add_argument("--image_size", type=int, help="The size of image", default=160)
    parser.add_argument("--max_num_images", type=int, help="The max number of images used to training, valid and test",
                        default=10000)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument("--logs_base_dir", type=str, help='Directory where to write event logs.', default='logs/')
    parser.add_argument("--models_base_dir", type=str, help="Direcotry where to save the parameters of model",
                        default="models/")


    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
