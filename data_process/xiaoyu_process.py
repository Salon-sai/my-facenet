# -*- coding: utf-8 -*-

import os

import numpy as np

class ImageAgeDatabase(object):

    def __init__(self, aligned_dir, num_folds=10, shuffle=False):
        self._aligned_dir = os.path.expanduser(aligned_dir)
        self._images_path = []
        self._labels = []
        self._embeddings = None
        self._num_folds = num_folds

        for root, dirs, files in os.walk(self._aligned_dir):
            if len(dirs) == 0:
                for file in files:
                    self._images_path.append(os.path.join(self._aligned_dir, root, file))
                self._labels += [int(root.split("/")[-1])] * len(files)

        self._images_path = np.asarray(self._images_path)
        self._labels = np.asarray(self._labels)
        self._current_fold = 0

        self._child_index = np.where(self._labels == 0)[0]
        self._middle_index = np.where(self._labels == 1)[0]
        self._old_index = np.where(self._labels == 2)[0]

        self._num_children = len(self._child_index)
        self._child_fold_size = int(np.ceil(self._num_children / num_folds))

        self._num_middle = len(self._middle_index)
        self._middle_fold_size = int(np.ceil(self._num_middle / num_folds))

        self._num_old = len(self._old_index)
        self._old_fold_size = int(np.ceil(self._num_old / num_folds))

        if shuffle:
            np.random.shuffle(self._child_index)
            np.random.shuffle(self._middle_index)
            np.random.shuffle(self._old_index)


    def split_index(self):
        children_train_index, children_valid_index = self._split_sub_index(
            fold_size=self._child_fold_size,
            num_data=self._num_children,
            data=self._child_index
        )
        middle_train_index, middle_valid_index = self._split_sub_index(
            fold_size=self._middle_fold_size,
            num_data=self._num_middle,
            data=self._middle_index
        )
        old_train_index, old_valid_index = self._split_sub_index(
            fold_size=self._old_fold_size,
            num_data=self._num_old,
            data=self._old_index
        )

        self._current_fold = (self._current_fold + 1) % self._num_folds

        train_indexes = np.concatenate([children_train_index, middle_train_index, old_train_index])
        valid_indexes = np.concatenate([children_valid_index, middle_valid_index, old_valid_index])

        return train_indexes, valid_indexes

    def _split_sub_index(self, fold_size, num_data, data):
        index = np.arange(num_data)
        start = self._current_fold * fold_size
        end = min((self._current_fold + 1) * fold_size, num_data)
        lower_bound = index >= start
        upper_bound = index < end

        cv_region = lower_bound * upper_bound
        valid_index = index[cv_region]
        train_index = index[~cv_region]

        return data[train_index], data[valid_index]

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
