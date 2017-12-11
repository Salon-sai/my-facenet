# -*- coding: utf-8 -*-

import os
import re

import tensorflow as tf
from tensorflow.python.platform import gfile

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

