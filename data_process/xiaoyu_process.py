# -*- coding: utf-8 -*-

import os

import numpy as np

class ImageAgeDatabase(object):

    def __init__(self, aligned_dir):
        self._aligned_dir = os.path.expanduser(aligned_dir)
        self._images_path = []
        self._labels = []
        self._embeddings = None

        for root, dirs, files in os.walk(self._aligned_dir):
            if len(dirs) == 0:
                for file in files:
                    self._images_path.append(os.path.join(self._aligned_dir, root, file))
                self._labels += [int(root.split("/")[-1])] * len(files)

        self._images_path = np.asarray(self._images_path)
        self._labels = np.asarray(self._labels)

        self._child_index = np.where(self._labels == 0)[0]
        self._middle_index = np.where(self._labels == 1)[0]
        self._old_index = np.where(self._labels == 2)[0]

        np.random.shuffle(self._child_index)
        np.random.shuffle(self._middle_index)
        np.random.shuffle(self._old_index)

        self._test_valid_indexes = np.concatenate([
            self._child_index[: int(0.1 * len(self._child_index))],
            self._middle_index[: int(0.1 * len(self._middle_index))],
            self._old_index[: int(0.1 * len(self._old_index))]
        ])

        self._train_indexes = np.concatenate([
            self._child_index[int(0.1 * len(self._child_index)): ],
            self._middle_index[int(0.1 * len(self._middle_index)): ],
            self._old_index[int(0.1 * len(self._old_index)): ]
        ])

        # self._num_sample = len(self._images_path)
        # indexes = np.arange(self._num_sample)
        # np.random.shuffle(indexes)
        # split_i = int(self._num_sample * 0.1)
        # self._test_indexes = indexes[: split_i]
        # self._train_indexes = indexes[split_i: ]

    @property
    def train_data(self):
        return self.images_path[self._train_indexes], \
               self.labels[self._train_indexes], \
               self.embeddings[self._train_indexes]

    @property
    def test_data(self):
        return self.images_path[self._test_valid_indexes], \
               self.labels[self._test_valid_indexes], \
               self.embeddings[self._test_valid_indexes]

    @property
    def valid_data(self):
        return self.images_path[self._test_valid_indexes], \
               self.labels[self._test_valid_indexes], \
               self.embeddings[self._test_valid_indexes]

    @property
    def images_path(self):
        return self._images_path

    @property
    def labels(self):
        return self._labels

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, embeddings):
        self._embeddings = embeddings
