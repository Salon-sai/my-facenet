# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

import optimizer
from classifier import train, save_variables_and_metagraph

def gender_model(embeddings, weight_decay1, phase_train=True):

    def ResBlock(net, num_outputs, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
        with tf.variable_scope(scope, "Res-Block", [net], reuse=reuse):
            up = slim.fully_connected(net, num_outputs=num_outputs, scope="hidden_1")
            up = slim.fully_connected(up, num_outputs=num_outputs, activation_fn=None,
                                      normalizer_fn=None, scope="hidden_2")

            net = tf.identity(net + scale * up, name="residual_add")

            if activation_fn:
                net = tf.nn.relu(net)
        return net

    with tf.variable_scope("gender_model"):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            biases_initializer=tf.constant_initializer(),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={
                                'decay': 0.995,
                                'epsilon': 0.001,
                                'updates_collections': None,
                                'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
                            },
                            weights_regularizer=slim.l1_regularizer(weight_decay1)):
            with slim.arg_scope([slim.batch_norm], is_training=phase_train):
                net = slim.repeat(embeddings, 3, ResBlock, num_outputs=128)

                net = slim.fully_connected(net, num_outputs=64, scope="change-cells-1")
                net = slim.repeat(net, 3, ResBlock, num_outputs=64)

                net = slim.fully_connected(net, num_outputs=32, scope="change-cells-2")
                net = slim.repeat(net, 6, ResBlock, num_outputs=32)

                net = slim.fully_connected(net, num_outputs=16, scope="change-cells-3")
                net = slim.repeat(net, 8, ResBlock, num_outputs=16)

                net = slim.fully_connected(net, num_outputs=8, scope="change-cells-4")
                net = slim.repeat(net, 6, ResBlock, num_outputs=8)

                net = slim.fully_connected(net, num_outputs=4, scope="change-cells-5")
                net = slim.repeat(net, 3, ResBlock, num_outputs=4)

                # net = slim.fully_connected(embeddings, num_outputs=128, scope="hidden_1")
                # net = slim.fully_connected(net, num_outputs=128, scope="hidden_2")
                # net = slim.fully_connected(net, num_outputs=128, scope="hidden_3")
                #
                # net = slim.fully_connected(net, num_outputs=64, scope="hidden_4")
                # net = slim.fully_connected(net, num_outputs=64, scope="hidden_5")
                # net = slim.fully_connected(net, num_outputs=32, scope="hidden_6")
                # net = slim.fully_connected(net, num_outputs=32, scope="hidden_7")
                # net = slim.fully_connected(net, num_outputs=32, scope="hidden_8")
                #
                # net = slim.fully_connected(net, num_outputs=16, scope="hidden_9")
                # net = slim.fully_connected(net, num_outputs=16, scope="hidden_10")
                # net = slim.fully_connected(net, num_outputs=16, scope="hidden_11")
                # net = slim.fully_connected(net, num_outputs=16, scope="hidden_12")
                # net = slim.fully_connected(net, num_outputs=16, scope="hidden_13")
                # net = slim.fully_connected(net, num_outputs=16, scope="hidden_14")
                #
                # net = slim.fully_connected(net, num_outputs=8, scope="hidden_15")
                # net = slim.fully_connected(net, num_outputs=8, scope="hidden_16")
                # net = slim.fully_connected(net, num_outputs=8, scope="hidden_17")
                # net = slim.fully_connected(net, num_outputs=4, scope="hidden_18")
                # net = slim.fully_connected(net, num_outputs=4, scope="hidden_19")
                # net = slim.fully_connected(net, num_outputs=4, scope="hidden_20")
                # net = slim.fully_connected(net, num_outputs=2, activation_fn=None, normalizer_fn=None, scope="logits")

                net = slim.fully_connected(net, num_outputs=2, activation_fn=None, normalizer_fn=None, scope="logits")
    return net

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

        logits = tf.identity(logits, "logits")

        predict = tf.argmax(logits, 1, name="predict")

        correct = tf.equal(predict, labels_placeholder, name="correct")

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
                                   record_var=True)

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
            gender_evaluate(session, valid_embeddings, valid_genders, embeddings_placeholder, labels_placeholder,
                            phase_train_placeholder, gs, epoch, correct_sum, summary_writer)
            epoch += 1
        session.close()

def gender_evaluate(session, valid_embeddings, valid_genders, embeddings_placeholder, labels_placeholder,
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

