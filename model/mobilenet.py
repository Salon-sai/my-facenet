# -*- coding: UTF-8 -*-


import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None, **kwargs):
    with slim.arg_scope([slim.convolution2d, slim.fully_connected, slim.separable_convolution2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        biases_initializer=slim.init_ops.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        return mobile_net(inputs=images,
                          width_multiplier=kwargs.get("width_multiplier", 1),
                          is_training=phase_train,
                          bottleneck_layer_size=bottleneck_layer_size,
                          keep_probability=keep_probability,
                          reuse=reuse)

def mobile_net(inputs,
               width_multiplier=1,
               is_training=True,
               bottleneck_layer_size=128,
               reuse=None,
               keep_probability=0.8,
               scope="MobileNet"):

    def _depthwise_separable_conv(inputs,
                                  num_pwc_filters,
                                  width_multiplier,
                                  sc,
                                  downsample=False):
        """
        define the layer is consisted of depth-wise convolution and point-wise convolution
        More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).
        :param inputs: a tensor of size (batch_size, height, width, channels)
        :param num_pwc_filters: the size of output channels
        :param width_multiplier: the width multiplier is used to thin the model
        :param sc: the scope name of layer
        :param downsample: is need to downsample
        :return:
        """
        num_pwc_filters = round(num_pwc_filters * width_multiplier)

        _stride = 2 if downsample else 1

        depthwise_conv = slim.separable_convolution2d(inputs=inputs,
                                                      num_outputs=None,
                                                      kernel_size=[3, 3],
                                                      depth_multiplier=1,
                                                      stride=_stride,
                                                      scope=sc + "/depthwise_conv")

        bn = slim.batch_norm(depthwise_conv, scope=sc + "/dw_batch_norm")

        pointwise_conv = slim.convolution2d(inputs=bn,
                                            num_outputs=num_pwc_filters,
                                            kernel_size=[1, 1],
                                            stride=1,
                                            padding="SAME",
                                            scope=sc + "/pointwise_conv")

        bn = slim.batch_norm(pointwise_conv, scope=sc + "/pw_batch_norm")

        return bn


    with tf.variable_scope(scope, "MobileNet", [inputs], reuse=reuse) as scope:
        end_points_collection = scope.name + "_end_points"
        # the activation_fn is executed
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                activation_fn=tf.nn.relu,
                                decay=0.995,
                                epsilon=0.001,
                                updates_collections=None,
                                variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                                fused=True):
                net = slim.convolution2d(inputs, num_outputs=32, kernel_size=[3, 3], stride=2, padding='SAME', scope="conv_1")
                net = _depthwise_separable_conv(net, num_pwc_filters=64, width_multiplier=width_multiplier, sc="conv_dp_2")
                net = _depthwise_separable_conv(net, num_pwc_filters=128, width_multiplier=width_multiplier, downsample=True, sc="conv_dp_3")
                net = _depthwise_separable_conv(net, num_pwc_filters=128, width_multiplier=width_multiplier, sc="conv_dp_4")
                net = _depthwise_separable_conv(net, num_pwc_filters=256, width_multiplier=width_multiplier, downsample=True, sc="conv_dp_5")
                net = _depthwise_separable_conv(net, num_pwc_filters=256, width_multiplier=width_multiplier, sc="conv_dp_6")
                net = _depthwise_separable_conv(net, num_pwc_filters=512, width_multiplier=width_multiplier, downsample=True, sc="conv_dp_7")

                net = _depthwise_separable_conv(net, num_pwc_filters=512, width_multiplier=width_multiplier, sc="conv_dp_8")
                net = _depthwise_separable_conv(net, num_pwc_filters=512, width_multiplier=width_multiplier, sc="conv_dp_9")
                net = _depthwise_separable_conv(net, num_pwc_filters=512, width_multiplier=width_multiplier, sc="conv_dp_10")
                net = _depthwise_separable_conv(net, num_pwc_filters=512, width_multiplier=width_multiplier, sc="conv_dp_11")
                net = _depthwise_separable_conv(net, num_pwc_filters=512, width_multiplier=width_multiplier, sc="conv_dp_12")

                net = _depthwise_separable_conv(net, num_pwc_filters=1024, width_multiplier=width_multiplier, downsample=True, sc="conv_dp_13")
                net = _depthwise_separable_conv(net, num_pwc_filters=1024, width_multiplier=width_multiplier, sc="conv_dp_14")

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                with tf.variable_scope("Logits"):
                    end_points['PrePool'] = net

                    net = slim.avg_pool2d(net, kernel_size=[7, 7], scope="avg_pool_15")

                    net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
                    end_points['squeeze'] = net

                    net = slim.dropout(net, keep_prob=keep_probability, is_training=is_training, scope="Dropout")
                    end_points["PreLogitsFlatten"] = net

                    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope="Bottleneck", reuse=False)

    return net