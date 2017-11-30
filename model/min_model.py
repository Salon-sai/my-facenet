# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def block(net, is_training, activation_fn=tf.nn.relu, scope=None, reuse=None):
    with tf.variable_scope(scope, 'min_block', [net], reuse=reuse):
        net = slim.conv2d(net, 32, [3, 3], stride=1, padding="SAME", scope='conv1')
        net = slim.max_pool2d(net, [3, 3], stride=3, scope="maxpool1")
        net = slim.batch_norm(net, is_training=is_training)
        if activation_fn:
            net = activation_fn(net)
    return net

def convlutional_layer(data, kernel_size, bias_size, pooling_size):
    kernel = tf.get_variable("conv", kernel_size, initializer=tf.random_normal_initializer())
    bias = tf.get_variable("bias", bias_size, initializer=tf.random_normal_initializer())

    conv = tf.nn.conv2d(data, kernel, strides=[1, 1, 1, 1], padding='VALID')
    linear_output = tf.nn.relu(tf.add(conv, bias))
    pooling = tf.nn.max_pool(linear_output, ksize=pooling_size, strides=pooling_size, padding="SAME")

    return pooling

def full_connection(data, weights_size, biases_size):
    weights = tf.get_variable("weights", weights_size, initializer=tf.random_normal_initializer())
    bias = tf.get_variable("bias", biases_size, initializer=tf.random_normal_initializer())

    return tf.add(tf.matmul(data, weights), bias)

def slim_convNet(data, keep_probability, phase_train, bottleneck_layer_size):
    net = slim.conv2d(data, 32, [3, 3],stride=1, padding='SAME', scope='conv1')
    net = tf.nn.relu(net)
    net = slim.max_pool2d(net, [3, 3], stride=3, padding="SAME", scope="max_pool1")
    net = slim.conv2d(net, 64, [3, 3], stride=1, padding='SAME', scope='conv2')
    net = tf.nn.relu(net)
    net = slim.max_pool2d(net, [3, 3], stride=3, padding="SAME", scope="max_pool2")

    net = slim.flatten(net)
    net = slim.dropout(net, keep_prob=keep_probability, is_training=phase_train, scope="Droput")
    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope="full_connected")
    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope="Bottleneck", reuse=False)
    return net

def original_convNet(data, image_size, keep_probability, bottleneck_layer_size):
    kernel_shape1=[3, 3, 1, 32]
    bias_shape1=[32]
    pooling_size1 = [1, 2, 2, 1]

    kernel_shape2=[3, 3, 32, 64]
    bias_shape2=[64]
    pooling_size2 = [1, 2, 2, 1]

    flatten_size = np.ceil(image_size / 4)
    full_conn_w_shape = [flatten_size * 64, 1024]
    full_conn_b_shape = [1024]
    out_w_shape = [1024, bottleneck_layer_size]

    with tf.variable_scope("conv_layer_1") as layer1:
        layer1_output = convlutional_layer(
            data=data,
            kernel_size=kernel_shape1,
            bias_size=bias_shape1,
            pooling_size=pooling_size1
        )

    with tf.variable_scope("conv_layer_1") as layer1:
        layer2_output = convlutional_layer(
            data=layer1_output,
            kernel_size=kernel_shape2,
            bias_size=bias_shape2,
            pooling_size=pooling_size2
        )

def mini_facenet(data, keep_probability, phase_train, bottleneck_layer_size, reuse=None):
    with tf.variable_scope("mini_facenet", reuse=reuse):
        net = slim.repeat(data, 3, block, is_training=phase_train)
        with tf.variable_scope("Logits"):
            net = slim.flatten(net)
            net = slim.dropout(net, keep_prob=keep_probability, is_training=phase_train, scope='Dropout')
        net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False)
    return net


def inference(images, keep_probability, phase_train, bottleneck_layer_size, reuse=None, **kwargs):
    with tf.variable_scope("mini_facenet", reuse=reuse):
        net = slim.repeat(images, 3, block, is_training=phase_train)
        with tf.variable_scope("Logits"):
            net = slim.flatten(net)
            net = slim.dropout(net, keep_prob=keep_probability, is_training=phase_train, scope='Dropout')
        net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False)
    return net
    # with slim.arg_scope([slim.conv2d, slim.fully_connected],
    #                     weights_initializer=tf.truncated_normal_initializer(stddev=0.1)):
    #     return slim_convNet(images, keep_probability, phase_train, bottleneck_layer_size)

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
    losses = tf.get_collection('loesses')
    # 把当前的总损失函数与现在的相加
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar('loss/' + l.op.name + ' (raw)', l)
        tf.summary.scalar('loss/' + l.op.name, loss_averages.average(l))
    return loss_averages_op

def variable_summaries(var, name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name + '_mean', mean)
    tf.summary.scalar(name + '_max', tf.reduce_max(var))
    tf.summary.scalar(name + '_min', tf.reduce_min(var))
    tf.summary.histogram(name, var)

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
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
            variable_summaries(var, var.op.name)
            # tf.summary.histogram(var.op.name, var)

        for grad, var in grads:
            variable_summaries(grad, "gradients/" + var.op.name)
            # tf.summary.histogram("gradients/" + var.op.name, grad)

    # 记录各个variable的平均值
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 完成梯度更新才执行之后的代码
    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name="train")

    return train_op

