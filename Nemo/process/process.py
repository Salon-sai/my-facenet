# -*- coding: utf-8 -*-

import argparse
import sys
import os
import re
from shutil import copyfile

import tensorflow as tf
from tensorflow.python.platform import gfile

import numpy as np

from scipy import misc


def main(args):
    root_dir = os.path.expanduser(args.data_dir)
    model_dir = os.path.expanduser(args.model_dir)
    pp_root_dir = os.path.join(root_dir, "preprocess")
    p_root_dir = os.path.join(root_dir, "process")
    if not os.path.isdir(p_root_dir):
        os.mkdir(p_root_dir)
    nrof_images = 0
    with tf.Session() as session:
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())

        load_model(model_dir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")


        image_paths = []
        for (root, dirs, files) in os.walk(pp_root_dir):
            if len(dirs) == 0:
                image_paths += [os.path.join(root, file) for file in files]

        nrof_images = len(image_paths)

        idx_array = np.arange(nrof_images)
        emb_arry = np.zeros((nrof_images, 128))
        nrof_batch = int(np.ceil(nrof_images / args.batch_size))
        for i in range(nrof_batch):
            start_index = i * args.batch_size
            end_index = min((i + 1) * args.batch_size, nrof_images)
            images = load_data(image_paths[start_index: end_index], args.image_size)
            emb_arry[start_index: end_index, :] = session.run(embeddings, feed_dict={images_placeholder: images, phase_train_placeholder: False})


        face_array = []
        while len(idx_array) > 0:
            index = idx_array[0]
            base_emb = emb_arry[index]
            dist = np.sum(np.square(np.subtract(base_emb, emb_arry[idx_array, :])), 1)
            idx = np.where(dist < args.threshold)[0].tolist()
            face_array.append(idx_array[idx])

            idx_array = np.delete(idx_array, idx)

        for index, face_idces in enumerate(face_array):
            label_dir = os.path.join(p_root_dir, str(index))
            os.mkdir(label_dir)
            for face_index in face_idces:
                image_name = image_paths[face_index].split("/")[-1]
                copyfile(image_paths[face_index], os.path.join(label_dir, image_name))


def load_model(model):
    model_path = os.path.expanduser(model)
    if os.path.isfile(model_path):
        print("Model file is %s " % model_path)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    else:
        print("Model path is %s" % model_path)
        meta_file, ckpt_file = get_model_filenames(model_path)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_path, ckpt_file))

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [f for f in files if f.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    if len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)

    meta_file = meta_files[0]
    ckpt_files = [f for f in files if '.ckpt' in f]
    max_step = -1
    for f in ckpt_files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def load_data(image_paths, image_size):
    nrof_sample = len(image_paths)
    images = np.zeros((nrof_sample, image_size, image_size, 3))
    for i in range(nrof_sample):
        images[i] = misc.imread(image_paths[i], mode='RGB')
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("model_dir", type=str, help="the directory of facenet cnn model")
    parser.add_argument("data_dir", type=str, help="the directory of data set")
    parser.add_argument("--image_size", type=int, help="the size of image when using the cnn", default=160)
    parser.add_argument("--batch_size", type=int, help="the enqueue batch size", default=3)
    parser.add_argument("--threshold", type=float, help="Use to classify the different and the same face", default=0.11)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))