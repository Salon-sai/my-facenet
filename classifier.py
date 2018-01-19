# -*- coding: utf-8 -*-

import os
import time

import tensorflow as tf
import numpy as np

def train(session, train_embeddings, train_labels, embeddings_placeholder, labels_placeholder, phase_train_placeholder,
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
            labels_placeholder: train_labels[start_index: end_index]
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
        print("[%3d/%6d] Batch Loss: %1.4f, Learning rate: %1.7f" % (gs // nrof_train_batch, gs, batch_loss, lr))

    epoch_summary = tf.Summary()
    epoch_summary.value.add(tag="train_batch/loss_mean", simple_value=batch_losses.mean())
    summary_writer.add_summary(epoch_summary, gs)

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
