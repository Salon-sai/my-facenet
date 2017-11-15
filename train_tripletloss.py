# -*- coding: UTF-8 -*-
import argparse
import sys

import data_process as dp
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import data_flow_ops

def sample_people(dataset, people_per_batch, images_per_person):
    """
    从训练集合中采样，将训练的dataset变成一个image_path的数组，其长度为nrof_images
    也就是本批次需要的图片数量。为了区分图片之间的类别，我们需要加入num_per_class来确定
    每个类别有的图片数量
    :param dataset: 训练数据集
    :param people_per_batch: 每个批数据所需要的人脸类型的数量
    :param images_per_person: 每个人脸类型所需要的最大照片数量
    :return:
    """
    image_paths = []
    num_per_class = []
    nrof_images = people_per_batch * images_per_person
    nrof_classes = len(dataset)
    idx = np.arange(nrof_classes)
    np.random.shuffle(idx)

    i = 0
    while len(image_paths) < nrof_images:
        class_idx = idx[i]
        nrof_images_in_class = len(dataset[class_idx])
        images_idx = np.arange(nrof_images_in_class)
        np.random.shuffle(images_idx)
        nrof_images_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        image_paths += dataset[class_idx][:nrof_images_class]
        num_per_class.append(nrof_images_class)
        i += 1

    # 校验每个类别的采样图片数量是否等于num_per_class对应每个元素的大小
    count = 0
    for i, num in enumerate(num_per_class):
        standard_class_name = image_paths[count].split("/")[5]
        for image_path in image_paths[count: count + num]:
            try:
                assert standard_class_name == image_path.split("/")[5]
            except:
                print(standard_class_name, image_path.split("/")[5])
                print(image_paths[count: count + num])
                print(num, i, len(num_per_class))
        count += num
    return image_paths, num_per_class

def main(args):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name="phase_train")

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name="image_paths")

        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name="labels")

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(3,), (3,)],
                                    shared_name=None, name=None)

        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        nrof_preprocess_threads = 4
        images_and_labels = []
        # 进行图片预处理
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                # 随机裁剪图片大小，以适应默认训练图片大小
                image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                # 是否随机左右翻转图片，以便进行数据增广
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size)

        train_ds, valid_ds, test_ds = dp.get_lfw_dataset()

        with tf.Session() as session:
            train(args, session, train_ds, enqueue_op, image_paths_placeholder, labels_placeholder)

def train(args, session, dataset, enqueue_op,image_paths_placeholder, labels_placeholder):
    nrof_examples = args.people_per_batch * args.images_per_person
    image_paths, num_per_class = sample_people(dataset=dataset,
                  people_per_batch=args.people_per_batch,
                  images_per_person=args.images_per_person)
    # 仅仅对image_paths进行标记
    labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
    image_paths_array = np.reshape(image_paths, (-1, 3))

    session.run(enqueue_op, feed_dict={image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, help="Image size (height, width) in a pixels.", default=160)
    parser.add_argument("--random_flip", help="random horizontal flipping of training images.", action="store_true")
    parser.add_argument("--batch_size", type=int, help="training batch size", default=90)
    parser.add_argument("--people_per_batch", type=int, help="Number of people of batch", default=45)
    parser.add_argument("--images_per_person", type=int, help="Number of images of people", default=40)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
