# -*- coding: utf-8 -*-

import argparse
import sys
import os
import time

import tensorflow as tf
import numpy as np

import optimizer

from tensorflow.contrib import slim
import datetime

from data_process.imdb_process import ImageDatabase, calculate_embedding

def gender_model(embeddings, weight_decay1, phase_train=True):
    with tf.variable_scope("gender_model"):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            biases_initializer=tf.constant_initializer(),
                            # activation_fn=None,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={
                                'decay': 0.995,
                                'epsilon': 0.001,
                                'updates_collections': None,
                                'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
                            },
                            weights_regularizer=slim.l1_regularizer(weight_decay1)):
            with slim.arg_scope([slim.batch_norm], is_training=phase_train):
                net = slim.fully_connected(embeddings, num_outputs=64, scope="hidden_1")
                net = slim.fully_connected(net, num_outputs=32, scope="hidden_2")
                net = slim.fully_connected(net, num_outputs=16, scope="hidden_3")
                net = slim.fully_connected(net, num_outputs=8, scope="hidden_4")
                net = slim.fully_connected(net, num_outputs=4, scope="hidden_5")
                net = slim.fully_connected(net, num_outputs=2, activation_fn=None, scope="logits")
    return net

def main(args):
    imdb_dir = os.path.expanduser(args.imdb_aligned_root)
    image_database = ImageDatabase(imdb_dir, args.max_num_images)
    batch_size = args.batch_size

    subdir = datetime.datetime.strftime(datetime.datetime.now(), 'gender-%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)


    image_database.embeddings = calculate_embedding(facenet_model_path=args.model_path,
                                                    images_path=image_database.images_path,
                                                    batch_size=batch_size,
                                                    image_size=args.image_size)
    embedding_size = image_database.embeddings.shape[1]


    gender_classifier(embedding_size, args.weight_decay_l1, args.learning_rate, args.learning_rate_decay_step,
                      args.learning_rate_decay_factor, args.optimizer, args.epoch_size, args.batch_size,
                      log_dir, model_dir, subdir, image_database)

def gender_classifier(embedding_size, weight_decay_l1, learning_rate, learning_rate_decay_step,
                      learning_rate_decay_factor, optimizer_name, epoch_size, batch_size,
                      log_dir, model_dir, subdir, image_database):
    _, train_embeddings, _, train_genders = image_database.train_data
    _, valid_embeddings, _, valid_genders = image_database.valid_data
    _, test_embeddings, _, test_genders = image_database.test_data

    print("The training number of female: %d" % np.sum(train_genders == 0))
    print("The training number of male: %d" % np.sum(train_genders == 1))

    print("The valid number of female: %d" % np.sum(valid_genders == 0))
    print("The valid number of male: %d" % np.sum(valid_genders == 1))

    with tf.Graph().as_default() as graph:
        labels_placeholder = tf.placeholder(dtype=tf.int64, shape=[None], name="gender_label")

        embeddings_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, embedding_size],
                                                name="embeddings_placeholder")

        phase_train_placeholder = tf.placeholder(dtype=tf.bool, name="phase_gender_train")

        global_step = tf.Variable(0, trainable=False)

        logits = gender_model(embeddings_placeholder, weight_decay_l1, phase_train_placeholder)

        correct = tf.equal(tf.argmax(logits, 1), labels_placeholder)

        correct_sum = tf.reduce_sum(tf.cast(correct, "float"))

        # accuracy_tensor = tf.reduce_mean(tf.cast(correct, "float"))

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits,
                                                                       name="softmax_cross_entropy")
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_losses = tf.add_n([cross_entropy_mean] + regularization_losses)

        update_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "gender_model")

        learning_rate = tf.train.exponential_decay(learning_rate, global_step, learning_rate_decay_step,
                                                   learning_rate_decay_factor, True)
        tf.summary.scalar("learning_rate", learning_rate)

        train_op = optimizer.train(total_loss=total_losses,
                                   global_step=global_step,
                                   optimizer=optimizer_name,
                                   learning_rate=learning_rate,
                                   moving_average_decay=0.99,
                                   update_gradient_vars=update_vars,
                                   log_historgrams=True)

        saver = tf.train.Saver(update_vars, max_to_keep=3)

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(log_dir, graph)

        session = tf.Session(graph=graph)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        epoch = 0
        while epoch < epoch_size:
            gs = session.run(global_step, feed_dict=None)

            train(session, train_embeddings, train_genders, embeddings_placeholder, labels_placeholder,
                  phase_train_placeholder, global_step, total_losses, learning_rate, train_op, summary_op,
                  summary_writer, batch_size)

            print("saving the model parameters...")
            save_variables_and_metagraph(session, saver, model_dir, subdir, gs)

            print("evaluating...")
            evaluate(session, valid_embeddings, valid_genders, embeddings_placeholder, labels_placeholder,
                     phase_train_placeholder, gs, epoch, correct_sum, summary_writer)
            epoch += 1
        session.close()

def train(session, train_embeddings, train_genders, embeddings_placeholder, labels_placeholder, phase_train_placeholder,
          global_step, total_losses, learning_rate, train_op, summary_op, summary_writer, batch_size):
    nrof_train_samples = len(train_embeddings)
    nrof_train_batch = int(np.ceil(nrof_train_samples / batch_size))
    batch_losses = np.empty((nrof_train_batch), dtype=np.float32)

    for i in range(nrof_train_batch):
        start_time = time.time()
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_train_samples)
        feed_dict = {
            phase_train_placeholder: True,
            embeddings_placeholder: train_embeddings[start_index: end_index],
            labels_placeholder: train_genders[start_index: end_index]
        }
        batch_loss, _, gs, summary, lr = \
            session.run([total_losses, train_op, global_step, summary_op, learning_rate],
                        feed_dict=feed_dict)
        batch_speed = time.time() - start_time
        summary_writer.add_summary(summary, gs)
        temp_summary = tf.Summary()
        temp_summary.value.add(tag="train_batch/loss", simple_value=batch_loss)
        temp_summary.value.add(tag="train_batch/speed_time", simple_value=batch_speed)
        summary_writer.add_summary(temp_summary, gs)

        batch_losses[i] = batch_loss
        print("[%3d/%6d] Batch Loss: %1.4f, Learning rate: %1.4f" % (gs // nrof_train_batch, gs, batch_loss, lr))

    epoch_summary = tf.Summary()
    epoch_summary.value.add(tag="train_batch/loss_mean", simple_value=batch_losses.mean())
    summary_writer.add_summary(epoch_summary, gs)

def evaluate(session, valid_embeddings, valid_genders, embeddings_placeholder, labels_placeholder,
             phase_train_placeholder, global_step, epoch, correct_sum, summary_writer):
    summary = tf.Summary()

    male_indexes = np.where(valid_genders == 1)[0]
    female_indexes = np.where(valid_genders == 0)[0]

    nrof_male = len(male_indexes)
    nrof_female = len(female_indexes)

    tp = session.run(correct_sum, feed_dict={
        embeddings_placeholder: valid_embeddings[male_indexes],
        labels_placeholder: valid_genders[male_indexes],
        phase_train_placeholder: False
    })
    fn = nrof_male - tp
    tn = session.run(correct_sum, feed_dict={
        embeddings_placeholder: valid_embeddings[female_indexes],
        labels_placeholder: valid_genders[female_indexes],
        phase_train_placeholder: False
    })
    fp = nrof_female - tn
    accuracy = float(tp + tn) / len(valid_embeddings)

    tpr = float(tp) / float(nrof_male)
    fpr = float(fp) / float(nrof_female)
    tnr = float(tn) / float(nrof_female)
    fnr = float(fn) / float(nrof_male)
    summary.value.add(tag="evaluate/TPR", simple_value=tpr)
    summary.value.add(tag="evaluate/FPR", simple_value=fpr)
    summary.value.add(tag="evaluate/TNR", simple_value=tnr)
    summary.value.add(tag="evaluate/FNR", simple_value=fnr)
    summary.value.add(tag="evaluate/accuracy", simple_value=accuracy)
    summary_writer.add_summary(summary, global_step)
    print("\t\t Epoch: %3d, Valid Accuracy: %1.4f" % (epoch, accuracy))


def save_variables_and_metagraph(session, saver, model_dir, model_name, global_step):
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(session, checkpoint_path, global_step=global_step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print("Variables saved in %.2f seconds" % save_time_variables)
    metagraph_filename = os.path.join(model_dir, "model-%s.meta" % model_name)
    if not os.path.exists(metagraph_filename):
        print("Saving metagraph")
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)


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
    parser.add_argument("--learning_rate", type=float, help="training learning rate", default=0.1)
    parser.add_argument("--learning_rate_decay_step", type=int,
                        help="Number of global step between learning rate decay.", default=10000)
    parser.add_argument('--learning_rate_decay_factor', type=float, help='Learning rate decay factor.', default=0.9)
    parser.add_argument("--logs_base_dir", type=str, help='Directory where to write event logs.', default='logs/')
    parser.add_argument("--models_base_dir", type=str, help="Direcotry where to save the parameters of model",
                        default="models/")
    parser.add_argument("--weight_decay_l1", type=float, help="L1 weight regularization", default=0.0)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
