# -*- coding: UTF-8 -*-
import argparse
import sys

import data_process as dp
import numpy as np
import tensorflow as tf
import model_network

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

        learning_rate_placeholder = tf.placeholder(tf.float32, name="learning_rate")

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name="phase_train")

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name="image_paths")

        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name="labels")

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(3,), (3,)], # feature个数为3
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

        # the name of image_batch: batch_join:0
        # the name of labels_batch: batch_join:1
        # len(images_and_labels)个线程将启动，即是nrof_preprocess_threads
        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size)

        # the variable of image_batch name is image_batch:0
        image_batch = tf.identity(image_batch, 'image_batch')
        # the variable of image_batch name is input:0
        image_batch = tf.identity(image_batch, 'input')
        # the variable of labels_batch name is label_batch:0
        labels_batch = tf.identity(labels_batch, 'label_batch')

        train_ds, valid_ds, test_ds = dp.get_lfw_dataset()


        # 创建神经网络并得到prelogits向量
        prelogits = model_network.inference(images=image_batch,
                                               keep_probability=0.5,
                                               phase_train=phase_train_placeholder,
                                               bottleneck_layer_size=args.embedding_size)
        # 对prelogits进行L2正则化
        embeddings = tf.nn.l2_normalize(x=prelogits, dim=1, epsilon=1e-10, name='embeddings')

        # 把embeddings先变成(?, 3, 128)的tensors，再把其分解成多个(3, 128)tensor，
        # 每个anchor，positive，negative都是(?, 128), ?是batch_size
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args.embedding_size]), 3, 1)
        triplet_loss = model_network.triplet_loss(anchor, positive, negative, args.alpha)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.epoch_size * args.learning_rate_decay_epochs,
                                                   args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        train_op = model_network.train(total_loss, global_step, args.optimizer,
                                       learning_rate, args.moving_average_decay, tf.trainable_variables())

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        summary_op = tf.summary.merge_all()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
            session.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})

            # summary_writer = tf.summary.FileWriter()

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=session)

            train(args, session, train_ds, enqueue_op, image_paths_placeholder, labels_placeholder, labels_batch,
                  batch_size_placeholder, phase_train_placeholder, learning_rate_placeholder, embeddings)

def train(args, session, dataset, enqueue_op,image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, phase_train_placeholder, learning_rate_placeholder ,embeddings):
    # nrof_examlpes: 每个批次处理的图片总数量
    nrof_examples = args.people_per_batch * args.images_per_person
    image_paths, num_per_class = sample_people(dataset=dataset,
                  people_per_batch=args.people_per_batch,
                  images_per_person=args.images_per_person)
    # 仅仅对image_paths进行标记
    labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
    image_paths_array = np.reshape(image_paths, (-1, 3))

    # 进行入队处理，把image_paths和labels_array放入队列中
    session.run(enqueue_op, feed_dict={image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_examples, args.embedding_size))

    nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
    # 用于求出采样的每张照片的embedding向量
    for i in range(nrof_batches):
        # 由于使用队列出队形式产生data batch，但args.batch_size不一定能正除nrof_examples，所以需要
        # 在batch_size与剩余图片之间做取最小值，作为出队的batch_size
        batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
        # 计算出batch_size个embeddings向量的值
        emb, lab = session.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                      learning_rate_placeholder: args.learning_rate,
                                                                      phase_train_placeholder: True})
        emb_array[lab, :] = emb
        print("----------batch embeddings-------------")
        print(emb)


def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch):
    """
    
    :param embeddings:
    :param nrof_images_per_class:
    :param image_paths:
    :param people_per_batch:
    :return:
    """
    emd_start_index = 0


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, help="Image size (height, width) in a pixels.", default=160)
    parser.add_argument("--random_flip", help="random horizontal flipping of training images.", action="store_true")
    parser.add_argument("--batch_size", type=int, help="Number of images to process in a batch.", default=90)
    parser.add_argument("--epoch_size", type=int, help="trainging epoch size", default=1000)
    parser.add_argument("--people_per_batch", type=int, help="Number of people of batch", default=45)
    parser.add_argument("--images_per_person", type=int, help="Number of images of people", default=40)
    parser.add_argument("--embedding_size", type=int, help="embedding size", default=128)
    parser.add_argument("--alpha", type=float, help="Positive to negative triplet distance margin.", default=0.2)
    parser.add_argument("--learning_rate", type=float, help="training learning rate", default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int, help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float, help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'], help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--moving_average_decay', type=float, help='Exponential decay for tracking of training parameters.', default=0.9999)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
