# -*- coding: utf-8 -*-
import optimizer

import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim
from classifier import train, save_variables_and_metagraph

def age_model(embeddings, weight_decay1, phase_train=True):
    with tf.variable_scope("age_model"):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
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
                # net = slim.fully_connected(embeddings, num_outputs=64, scope="hidden_1")
                # net = slim.fully_connected(net, num_outputs=32, scope="hidden_2")
                # net = slim.fully_connected(net, num_outputs=16, scope="hidden_3")
                # net = slim.fully_connected(net, num_outputs=8, scope="hidden_4")
                net = slim.fully_connected(embeddings, num_outputs=3, activation_fn=None, normalizer_fn=None, scope="logits")
    return net

def age_classifier(embedding_size, weight_decay_l1, learning_rate, learning_rate_decay_step,
                   learning_rate_decay_factor, optimizer_name, epoch_size, batch_size, gpu_memory_fraction,
                   log_dir, model_dir, subdir, image_database, n_fold=10):
    # _, train_ages, train_embeddings = image_database.train_data
    # _, valid_ages, valid_embeddings = image_database.valid_data

    with tf.Graph().as_default() as graph:
        labels_placeholder = tf.placeholder(dtype=tf.int64, shape=[None], name="gender_label")

        embeddings_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, embedding_size],
                                                name="embeddings_placeholder")

        phase_train_placeholder = tf.placeholder(dtype=tf.bool, name="phase_gender_train")

        global_step = tf.Variable(0, trainable=False)

        logits = age_model(embeddings_placeholder, weight_decay_l1, phase_train_placeholder)

        correct = tf.equal(tf.argmax(logits, 1), labels_placeholder)

        correct_sum = tf.reduce_sum(tf.cast(correct, "float"))

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits,
                                                                       name="softmax_cross_entropy")
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_losses = tf.add_n([cross_entropy_mean] + regularization_losses)

        update_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "age_model")

        learning_rate = tf.train.exponential_decay(learning_rate, global_step, learning_rate_decay_step,
                                                   learning_rate_decay_factor, True)
        tf.summary.scalar("learning_rate", learning_rate)

        train_op = optimizer.train(total_loss=total_losses,
                                   global_step=global_step,
                                   optimizer=optimizer_name,
                                   learning_rate=learning_rate,
                                   moving_average_decay=0.99,
                                   update_gradient_vars=update_vars)

        saver = tf.train.Saver(update_vars, max_to_keep=3)

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(log_dir, graph)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        session = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options))

        accuracies = np.empty(n_fold)

        for i in range(n_fold):
            train_index, valid_index = image_database.split_index()
            train_embeddings = image_database.embeddings[train_index]
            train_ages = image_database.labels[train_index]

            valid_embeddings = image_database.embeddings[valid_index]
            valid_ages = image_database.labels[valid_index]

            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())

            epoch = 0
            last_accuracy = 0
            # youth_indexes = np.where(train_ages <= 30)[0]
            # middle_indexes = np.where(np.logical_and(train_ages > 30, train_ages <= 59))[0]
            # old_indexes = np.where(train_ages > 59)[0]
            while epoch < epoch_size:
                gs = session.run(global_step, feed_dict=None)

                # selection_ages, selection_embeddings = ages_selection(train_embeddings,
                #                                                       youth_indexes,
                #                                                       middle_indexes,
                #                                                       old_indexes)

                train(session, train_embeddings, train_ages, embeddings_placeholder, labels_placeholder,
                      phase_train_placeholder, global_step, total_losses, learning_rate, train_op, summary_op,
                      summary_writer, batch_size)

                print("saving the model parameters...")
                save_variables_and_metagraph(session, saver, model_dir, subdir, gs)

                print("evaluating...")
                last_accuracy = age_evaluate(session, valid_embeddings, valid_ages, embeddings_placeholder, labels_placeholder,
                             phase_train_placeholder, gs, epoch, correct_sum, summary_writer)

                epoch += 1

            accuracies[i] = last_accuracy

        print("After %d-Fold Cross Validation")
        print("Mean Accuracy: %1.4f+-%1.4f" % (accuracies.mean(), accuracies.std()))

        session.close()

def ages_selection(embeddings, youth_indexes, middle_indexes, old_indexes, max_num=None):
    np.random.shuffle(youth_indexes)
    np.random.shuffle(middle_indexes)
    np.random.shuffle(old_indexes)

    if max_num is None:
        num_image = min(len(youth_indexes), len(middle_indexes), len(old_indexes))
    else:
        num_image = min(len(youth_indexes), len(middle_indexes), len(old_indexes), max_num)

    selection_ages = np.concatenate((np.full((num_image), 0),
                    np.full((num_image), 1),
                    np.full((num_image), 2)))
    selection_embeddings = np.concatenate((embeddings[youth_indexes[:num_image]],
                                           embeddings[middle_indexes[:num_image]],
                                           embeddings[old_indexes[:num_image]]), axis=0)

    return selection_ages, selection_embeddings


def age_evaluate(session, valid_embeddings, valid_ages, embeddings_placeholder, labels_placeholder,
                 phase_train_placeholder, global_step, epoch, correct_sum, summary_writer):
    summary = tf.Summary()

    # youth_index = np.where(valid_ages <= 30)[0]
    # middle_index = np.where(np.logical_and(valid_ages > 30, valid_ages <= 59))[0]
    # old_index = np.where(valid_ages > 59)[0]
    #
    # age_labels = np.empty((len(valid_ages)))
    #
    # age_labels[youth_index] = 0
    # age_labels[middle_index] = 1
    # age_labels[old_index] = 2

    correct_count = session.run(correct_sum, feed_dict={
        embeddings_placeholder: valid_embeddings,
        labels_placeholder: valid_ages,
        phase_train_placeholder: False
    })

    accuracy = float(correct_count) / float(len(valid_ages))

    summary.value.add(tag="evaluate/accuracy", simple_value=accuracy)
    summary_writer.add_summary(summary, global_step)
    print("\t\t Epoch: %3d, Valid Accuracy: %1.4f" % (epoch, accuracy))
    return accuracy
