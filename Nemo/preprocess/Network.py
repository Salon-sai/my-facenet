# -*- coding: utf-8 -*-

from six import iteritems, string_types

import os

import numpy as np
import tensorflow as tf


def layer(op):

    def layer_decorated(self, *args, **kwargs):
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))

        if len(self.terminals) == 0:
            raise  RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)

        # Perform the operation and get the output
        layer_output = op(self, layer_input, *args, **kwargs)

        self.layers[name] = layer_output

        self.feed(layer_output)

        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True):
        self.input = inputs
        self.terminals = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()


    def setup(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        """
        加载模型权重参数
        :param data_path: 模型参数的路径（numpy-serialized）
        :param session: 当前的tensorflow session
        :param ignore_missing:
        :return:
        """

        data_dict = np.load(data_path, encoding='latin1').item()

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assgin(data))
                    except ValueError:
                        if not ignore_missing:
                            raise


    def feed(self, *args):

        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)

            self.terminals.append(fed_layer)

        return self


    def get_output(self):
        return self.terminals[-1]


    def get_unique_name(self, prefix):

        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return "%s_%d" % (prefix, ident)

    def make_var(self, name, shape):
        """
        创建variable
        :param name: variable的名字
        :param shape: variable的大小
        :return:
        """
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):

        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=None, padding='SAME', group=1, biased=True):
        self.validate_padding(padding)
        # 获取input的channels
        c_i = int(input.get_shape()[-1])

        assert c_i % group == 0
        assert c_o % group == 0

        convolve = lambda i, k: tf.nn.conv2d(i, k [1, s_h, s_w, 1], padding=padding)

        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group , c_o])
            output = convolve(input, kernel)

            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)

            if relu:
                output = tf.nn.relu(output, name=scope.name)

            return output

    @layer
    def prelu(self, input, name):
        """p-RELU激活函数"""
        with tf.variable_scope(name):
            i = int(input.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            output = tf.nn.relu(input) + tf.multiply(alpha, -tf.nn.relu(input))

        return output


    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding='SAME'):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name):
            input_shape = input.get_shape
            # 将输入的数据拉直为一个向量
            dim = 1

            if input_shape.ndims == 4:
                for d in input_shape[1:].as_list():
                    dim *= int(d)

                feed_in = tf.reshape(input_shape, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)

            weights = self.make_var("weights", shape=[dim, num_out])
            biases = self.make_var("biases", shape=[num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=name)
            return fc

    @layer
    def softmax(self, target, axis, name=None):
          max_axis = tf.reduce_max(target, axis, keep_dims=True)
          target_exp = tf.exp(target - max_axis)
          normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
          softmax = tf.div(target_exp, normalize, name)
          return softmax


class PNet(Network):

    def setup(self):
        (self.feed('data')
            .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
            .prelu(name='PReLU1')
            .max_pool(2, 2, 2, 2, name='pool1')
            .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
            .prelu(name='PReLU2')
            .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
            .conv(2, 2, 64, 1, 1, padding="VALID", relu=False, name='conv3')
            .prelu('prelu3')
            .fc(128, relu=False, name='conv4')
            .prelu(name='prelu4')
            .fc(2, relu=False, name='conv5-1')
            .softmax(1, name='prob1'))

        (self.feed('prelu4')
             .fc(4, relu=False, name='conv5-2'))


class RNet(Network):

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 28, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 48, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .fc(128, relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(2, relu=False, name='conv5-1')
             .softmax(1, name='prob1'))

        (self.feed('prelu4')
             .fc(4, relu=False, name='conv5-2'))


class ONet(Network):

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(256, relu=False, name='conv5')
             .prelu(name='prelu5')
             .fc(2, relu=False, name='conv6-1')
             .softmax(1, name='prob1'))

        (self.feed('prelu5')
             .fc(4, relu=False, name='conv6-2'))

        (self.feed('prelu5')
             .fc(10, relu=False, name='conv6-3'))



def create_mtcnn(session, model_path):
    if not model_path:
        model_path, _ = os.path.split(os.path.realpath(__file__))

    with tf.variable_scope('pnet'):
        data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
        pnet = PNet({'data': data})
        pnet.load(os.path.join(model_path, 'det1.npy'), session)


    with tf.variable_scope('rnet'):
        data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
        rnet = RNet({'data': data})
        rnet.load(os.path.join(model_path, 'det2.npy'), session)

    with tf.variable_scope('rnet'):
        data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
        onet = ONet({'data': data})
        onet.load(os.path.join(model_path, 'det3.npy'), session)


    pnet_fun = lambda img : session.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0': img})
    rnet_fun = lambda img : session.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0': img})
    onet_fun = lambda img : session.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0': img})

    return pnet_fun, rnet_fun, onet_fun















