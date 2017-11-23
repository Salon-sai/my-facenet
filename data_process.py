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
    generate_evaluate_dataset(valid_set)

def get_lfw_dataset():
    dataset = get_dataset("~/data/lfw")
    return split_dataset(dataset=dataset)

def generate_evaluate_dataset(dataset):
    """
    生成用于校验的数据集合
    :param dataset: 原始的数据集合，每个元素都是同一个人的人脸图像路径数组
    :return:
    """
    evaluate_dataset = []
    issame_array = []
    num_people = len(dataset)
    for index, per_person_images in enumerate(dataset):
        num_images = len(per_person_images)
        if num_images > 1:
            for i in range(num_images):
                for j in range(i + 1, num_images):
                    evaluate_dataset.append((per_person_images[i], per_person_images[j]))
                    issame_array.append(True)
        elif num_images == 1:
            other_index = np.random.choice(np.where(num_people != index)[0])
            other_person_images = dataset[other_index]
            if len(other_person_images) == 1:
                other_image = other_person_images[0]
            else:
                other_image = np.random.choice(other_person_images)
            evaluate_dataset.append((per_person_images[0], other_image))
            issame_array.append(False)
    # 为了满足之后reshape(-1, 3)，所以需要将数据集进行裁剪
    actual_len = ((len(evaluate_dataset) * 2 // 3) * 3) // 2
    evaluate_dataset = evaluate_dataset[:actual_len]
    return evaluate_dataset, issame_array

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help="Directory with training data set", default="~/data/lfw")
    return parser.parse_args(argv)

# if __name__ == '__main__':
#     main(parse_arguments(sys.argv[1:]))