# -*- coding: utf-8 -*-

import tensorflow as tf

def triplet_loss(anchor, positive, negative, alpha):
    """
    计算triplet loss根据FaceNet的论文
    :param anchor: 标准图片的embedding向量
    :param positive: 与标准图片同一类别的embedding向量
    :param negative: 与标准图片不同类别的embedding向量
    :param alpha: 正反图片embedding之间的间距
    :return:
    """
    with tf.variable_scope("triplet_loss"):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        basis_loss = tf.add(tf.subtract(pos_dist, neg_dist), 1)
        loss = tf.reduce_sum(tf.maximum(basis_loss, 0), 0)
    return loss


def _add_loss_summaries(total_loss):
    """

    :param total_loss:
    :return:
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar('loss/' + l.op.name + ' (raw)', l)
        tf.summary.scalar('loss/' + l.op.name, loss_averages.average(l))
    return loss_averages_op

def variable_summaries(var, name):
    """

    :param var:
    :param name:
    :return:
    """
    mean = tf.reduce_sum(var)
    tf.summary.scalar(name + "_mean", mean)
    tf.summary.scalar(name + "_max", tf.reduce_max(var))
    tf.summary.scalar(name + "_min", tf.reduce_min(var))
    tf.summary.histogram(name, var)

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_historgrams=True):

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if log_historgrams:
        for var in tf.trainable_variables():
            variable_summaries(var, var.name)

        for grad, var in grads:
            variable_summaries(grad, "gradient/" + var.op.name)

    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name="train")

    return train_op
