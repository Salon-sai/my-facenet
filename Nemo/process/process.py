# -*- coding: utf-8 -*-

import argparse
import sys
import os

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import numpy as np

def main(args):
    root_dir = os.path.expanduser(args.data_dir)
    pp_root_dir = os.path.join(root_dir, "perprocess")
    p_root_dir = os.path.join(root_dir, "process")

    batch_size_placeholder = tf.placeholder(tf.int32, name="batch_size")

    image_paths_placeholder = tf.placeholder(tf.string, name="image_paths")

    input_queue = data_flow_ops.FIFOQueue(capacity=100000, dtypes=[tf.string], shared_name=None, name=None)

    enqueue_op = input_queue.enqueue_many([image_paths_placeholder])

    images_and_paths = []

    for _ in range(4):
        filename = input_queue.dequeue()
        image = image_pre_process(args.image_size, filename)
        images_and_paths.append([image, filename])


    images_batch, paths_batch = tf.train.batch_join(images_and_paths, batch_size=batch_size_placeholder,
                                                    shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
                                                    capacity=4 * 4 * args.batch_size, allow_smaller_final_batch=True)


def image_pre_process(image_size, filename):
    file_contents = tf.read_file(filename)
    image = tf.image.decode_image(file_contents, channels=3)
    image = tf.random_crop(image, [image_size, image_size, 3])
    image.set_shape((image_size, image_size, 3))
    return image

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("model_dir", type=str, help="the directory of facenet cnn model")
    parser.add_argument("data_dir", type=str, help="the directory of data set")
    parser.add_argument("--image_size", type=int, help="the size of image when using the cnn", default=224)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
