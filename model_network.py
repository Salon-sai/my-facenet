# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim

def block(net, is_training, activation_fn=tf.nn.relu, scope=None, reuse=None):
    with tf.variable_scope(scope, 'min_block', [net], reuse=reuse):
        net = slim.conv2d(net, 32, [3, 3], stride=1, padding="SAME", scope='conv1')
        net = slim.max_pool2d(net, [3, 3], stride=1, scope="maxpool1")
        net = slim.batch_norm(net, is_training=is_training)
        if activation_fn:
            net = activation_fn(net)
    return net

def inference(images, keep_probability, phase_train, bottleneck_layer_size, reuse=None):
    with tf.variable_scope("mini_facenet", reuse=reuse):
        net = slim.repeat(images, 3, block, is_training=phase_train)
        with tf.variable_scope("Logits"):
            net = slim.flatten(net)
            net = slim.dropout(net, keep_prob=keep_probability, is_training=phase_train, scope='Dropout')
        net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False)
    return net

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
        baisc_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(baisc_loss, 0), 0)
    return loss

def _add_loss_summaries(total_loss):
    """

    :param total_loss:
    :return:
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    # 获取之前在命名为losses的variable
    losses = tf.get_collection("losses")
    # 把当前的总损失函数与现在的相加
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):

    # loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([total_loss]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # 对需要训练的变量记录其直方图
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        for grad, var in grads:
            tf.summary.histogram(var.op.name + "/gradients", grad)

    # 记录各个variable的平均值
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 完成梯度更新才执行之后的代码
    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name="train")

    return train_op

