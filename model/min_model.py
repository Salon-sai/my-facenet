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