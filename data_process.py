# -*- coding: UTF-8 -*-

import argparse
import os
import sys
import numpy as np

def get_dataset(paths):
    dataset = []
    for path in paths.split(":"):
        path_exp = os.path.expanduser(path)
        class_names = os.listdir(path_exp)
        class_names.sort()
        for class_name in class_names:
            class_dir = os.path.join(path_exp, class_name)
            # 每个类别一个image_paths数组，因此len(dataset)为类别的总数
            image_paths = get_image_path(class_dir)
            dataset.append(image_paths)
    return dataset

def get_image_path(class_dir):
    image_paths = []
    if os.path.isdir(class_dir):
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if os.path.isfile(file_path):
                image_paths.append(file_path)
    return image_paths

def split_dataset(dataset):
    # split the dataset to training set, valid set, test set
    one_image_cls_indices = []
    more_images_cls_indices = []

    for index in range(len(dataset)):
        if len(dataset[index]) == 1:
            one_image_cls_indices.append(index)
        elif len(dataset[index]) > 1:
            more_images_cls_indices.append(index)

    np.random.shuffle(one_image_cls_indices)
    one_image_cls_num = len(one_image_cls_indices)
    smallpart_one_image_num = int(np.ceil(one_image_cls_num * 0.1))

    np.random.shuffle(more_images_cls_indices)
    more_images_cls_num = len(more_images_cls_indices)
    smallpart_more_images_num = int(np.ceil(more_images_cls_num * 0.1))

    test_idx = one_image_cls_indices[:smallpart_one_image_num] + more_images_cls_indices[:smallpart_more_images_num]
    valid_idx = one_image_cls_indices[smallpart_one_image_num:2*smallpart_one_image_num] + more_images_cls_indices[smallpart_more_images_num:2*smallpart_more_images_num]
    train_idx = one_image_cls_indices[2*smallpart_one_image_num:] + more_images_cls_indices[2*smallpart_more_images_num:]

    np.random.shuffle(test_idx)
    np.random.shuffle(valid_idx)
    np.random.shuffle(train_idx)

    all_idx = test_idx + valid_idx + train_idx
    assert len(all_idx) == len(dataset)

    train_set = [dataset[i] for i in train_idx]
    valid_set = [dataset[i] for i in valid_idx]
    test_set = [dataset[i] for i in test_idx]

    return train_set, valid_set, test_set

def main(args):
    dataset = get_dataset(args.input_dir)
    train_set, valid_set, test_set = split_dataset(dataset)

def get_lfw_dataset():
    dataset = get_dataset("~/data/lfw")
    return split_dataset(dataset=dataset)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help="Directory with training data set")
    return parser.parse_args(argv)

# if __name__ == '__main__':
#     main(parse_arguments(sys.argv[1:]))