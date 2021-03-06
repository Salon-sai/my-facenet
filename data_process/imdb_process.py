# -*- coding: utf-8 -*-

import os

import numpy as np
import tensorflow as tf
from model import utils
from data_process.utils import load_data

class ImageDatabase(object):

    def __init__(self, aligned_dir, max_num_images):
        self._aligned_dir = aligned_dir
        self._load_aligned_images_path(max_num_images)
        self._split_with_index()

        self._embeddings = None
        self.nrof_images = len(self._images_path)

    def _load_aligned_images_path(self, max_num_images):
        images_path = []
        genders = []
        ages = []
        for i in os.listdir(self._aligned_dir):
            _path = os.path.join(self._aligned_dir, i)
            if not os.path.isdir(_path):
                continue
            for image_name in os.listdir(_path):
                image_path = os.path.join(_path, image_name)
                _splits = image_name.split("_")
                age = int(_splits[-3])
                gender = int(_splits[-2])
                genders.append(gender)
                ages.append(age)
                images_path.append(image_path)
            if len(images_path) > max_num_images:
                break
        self._images_path = np.asarray(images_path)
        self._genders = np.asarray(genders)
        self._ages = np.asarray(ages)

    def _split_with_index(self):
        nrof_images = len(self.images_path)
        indexes = np.arange(nrof_images)
        np.random.shuffle(indexes)
        split_i = int(np.ceil(nrof_images * 0.1))
        self._valid_indexes = indexes[:split_i]
        self._test_indexes = indexes[split_i: 2 * split_i]
        self._train_indexes = indexes[2 * split_i:]

    @property
    def valid_data(self):
        return self._images_path[self._valid_indexes], \
               self._embeddings[self._valid_indexes], \
               self._ages[self._valid_indexes], \
               self._genders[self._valid_indexes], \

    @property
    def test_data(self):
        return self._images_path[self._test_indexes], \
               self._embeddings[self._test_indexes], \
               self._ages[self._test_indexes], \
               self._genders[self._test_indexes]

    @property
    def train_data(self):
        return self._images_path[self._train_indexes], \
               self._embeddings[self._train_indexes], \
               self._ages[self._train_indexes], \
               self._genders[self._train_indexes]

    @property
    def images_path(self):
        return self._images_path

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, embeddings):
        self._embeddings = embeddings


def calculate_embedding(facenet_model_path, images_path, batch_size, image_size):
    nrof_images = len(images_path)

    with tf.Session() as session:
        utils.load_model(model=facenet_model_path)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        embedding_size = embeddings.get_shape()[1]

        emb_array = np.zeros((nrof_images, embedding_size))

        nrof_batch = int(np.ceil(nrof_images / batch_size))

        for i in range(nrof_batch):
            start_index = i * batch_size
            end_index = min(nrof_images, (i + 1) * batch_size)
            images = load_data(images_path[start_index: end_index], False, False, image_size)
            feed_dict = { images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index: end_index] = session.run(embeddings, feed_dict=feed_dict)

    return emb_array