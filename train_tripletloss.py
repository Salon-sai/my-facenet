# -*- coding: UTF-8 -*-
import argparse
import sys
import itertools
import time
import datetime
import os

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
    image_paths = []    # 本次采样中，所有的样本图片的路径
    num_per_class = []  # 本次采样中，每个类别的样本数目
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
    subdir = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

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
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)

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

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        with tf.Session() as session:
            session.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
            session.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})

            summary_writer = tf.summary.FileWriter(logdir=log_dir, graph=session.graph)

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=session)

            epoch = 0
            # 开始进行回合训练
            while epoch < args.max_nrof_epochs:
                gs = session.run(global_step, feed_dict=None)
                # 因为每次训练都会执行若干个step，而且step并不确定。
                # 所以需要使用整除方式判断时候满足一个回合所需的step
                epoch = gs // args.epoch_size

                train(args, session, train_ds, epoch, enqueue_op, image_paths_placeholder, labels_placeholder, labels_batch,
                      batch_size_placeholder, phase_train_placeholder, learning_rate_placeholder,
                      embeddings, total_loss, train_op, global_step, learning_rate,
                      summary_writer)

def train(args, session, dataset, epoch, enqueue_op,image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, phase_train_placeholder, learning_rate_placeholder ,
          embeddings, loss, train_op, global_step, learning_rate,
          summary_writter):

    if args.learning_rate < 0:
        pass

    # 记录本次回合已经训练的step次数
    batch_number = 0
    gs = 0

    while batch_number < args.epoch_size:
        image_paths, num_per_class = sample_people(dataset=dataset,
                      people_per_batch=args.people_per_batch,
                      images_per_person=args.images_per_person)

        start_time = time.time()
        # nrof_examlpes: 每个批次处理的图片总数量
        nrof_examples = args.people_per_batch * args.images_per_person
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
            try:
                emb, lab = session.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                              learning_rate_placeholder: args.learning_rate,
                                                                              phase_train_placeholder: True})
                emb_array[lab, :] = emb
            except tf.errors.OutOfRangeError:
                print("enqueue number of image", len(image_paths))


        triplets, num_triplets = select_triplets(embeddings=emb_array,
                        nrof_images_per_class=num_per_class,
                        image_paths=image_paths,
                        people_per_batch=args.people_per_batch,
                        alpha=args.alpha)
        selection_time = time.time() - start_time
        print("select triplets speed the time: %.3f s" % (time.time() - start_time))
        nrof_batches = int(np.ceil(3 * num_triplets / args.batch_size))
        triplets_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplets_paths)), (-1, 3))
        triplets_path_array = np.reshape(np.expand_dims(np.array(triplets_paths), 1), (-1, 3))
        # 进行入队操作
        # 按照(anchor, positive, negative)图片三元组作为训练一个训练样本
        session.run(enqueue_op, feed_dict={image_paths_placeholder: triplets_path_array, labels_placeholder: labels_array})
        nrof_examples = len(triplets_paths)
        print("3 times num_triplets: %d nrof_examples: %d" % (3 * num_triplets, nrof_examples))

        emb_array = np.zeros((nrof_examples, args.embedding_size))
        loss_array = np.zeros(num_triplets)
        train_time = 0
        for i in range(nrof_batches):
            start_time = time.time()
            batch_size = min(args.batch_size, nrof_examples - i * args.batch_size)
            print("batch_size %d\t learning_rate %.4f" % (batch_size, args.learning_rate))
            # 执行训练操作，给神经网络喂养batch_size个样本
            l, _, gs, lr, emb, lab = session.run([loss, train_op, global_step, learning_rate, embeddings, labels_batch],
                        feed_dict={
                            phase_train_placeholder: True,
                            batch_size_placeholder: batch_size,
                            learning_rate_placeholder: args.learning_rate,
                        })
            # 记录并保存learning rate
            args.learning_rate = lr
            emb_array[lab, :] = emb
            loss_array[i] = l
            duration = time.time() - start_time
            print("Epoch [%d][%d/%d]\t Global Time %d \t Time %.3f\t Loss %2.3f"
                  % (epoch, batch_number + 1, args.batch_size, gs, duration, l))
            train_time += duration
            batch_number += 1
        print("The %d-th epoch spend %.4f s training" % (epoch, train_time))
        summary = tf.Summary()
        summary.value.add(tag="time/selection", simple_value=selection_time)
        summary_writter.add_summary(summary, gs)

    return gs


def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """
    根据embedding之间的相似度来选择偏离最大的negative和positive样本
    :param embeddings: 该批数据的embedding向量
    :param nrof_images_per_class: 每一个类别的样本数量
    :param image_paths: 每个图片样本的路径
    :param people_per_batch: 本次batch的类别数目（人脸数目）
    :return:
    """
    # 记录同一类别图片的开始index
    emd_start_index = 0
    num_trips = 0  # 预期triplet的数量
    triplets = []

    for image_per_class in nrof_images_per_class:
        # 记录同一类别图片的结束index
        emd_end_index = emd_start_index + int(image_per_class)
        # 同一类别图片的所有embedding向量
        image_per_embeddings = embeddings[emd_start_index: emd_end_index]
        for j in range(image_per_class):
            anchor_i = emd_start_index + j
            neg_dists = np.sum(np.square(embeddings - embeddings[anchor_i]), 1)
            neg_dists[emd_start_index: emd_end_index] = np.NAN
            for pair in range(j + 1, image_per_class):
                # 每一对positive embedding作为一个positive样本
                pos_i = emd_start_index + pair
                pos_dist = np.sum(np.square(embeddings[anchor_i] - embeddings[pos_i]))
                all_neg_i = np.where(neg_dists - pos_dist < alpha)[0]
                nrof_random_negs = len(all_neg_i)
                if nrof_random_negs > 0:
                    rnd_idx = np.argmin(neg_dists[all_neg_i])
                    # rnd_idx = np.random.randint(nrof_random_negs) # 源代码通过随机选取neg的idx，而不是选择最小的那个neg_dist
                    neg_i = all_neg_i[rnd_idx]
                    # print("positive dist: %1.4f negative dist %1.4f" % (pos_dist, neg_dists[neg_i]))
                    triplets.append((image_paths[anchor_i], image_paths[pos_i], image_paths[neg_i]))
                num_trips += 1

        emd_start_index = emd_end_index
    # print(np.sum(nrof_images_per_class[np.where(nrof_images_per_class > 1)] - len(np.where(nrof_images_per_class > 1))))
    np.random.shuffle(triplets)
    print("number of triplets : %d, expect number of triplets: : %d" % (len(triplets), num_trips))
    return triplets, len(triplets)

def save_variabels_and_metagraph(session, saver, summary_writter, model_dir, model_name, step):
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(session, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print("Variables saved in %.2f seconds" % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2d seconds' % save_time_metagraph)

def evaluate(session, dataset, embeddings, labels_batch, enqueue_op,
             image_paths_placeholder, labels_placeholder, batch_size_placeholder, phase_train_placeholder, learning_rate_placeholder,
             args):
    validate_dataset, issame_array = dp.generate_evaluate_dataset(dataset)
    nrof_images = len(validate_dataset) * 2
    labels_array = np.reshape(np.arange(nrof_images), (-1, 3))
    image_paths_array = np.reshape(np.expand_dims(np.array(validate_dataset), 1), (-1, 3))
    assert labels_array.shape == image_paths_array.shape
    session.run(enqueue_op, feed_dict={image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, args.embedding_size))
    nrof_batch = int(np.ceil(nrof_images / args.batch_size))

    for i in range(nrof_batch):
        batch_size = min(args.batch_size, nrof_images - i * args.batch_size)
        emb, lab = session.run([embeddings, labels_batch],
                               feed_dict={
                                   batch_size_placeholder: batch_size,
                                   phase_train_placeholder: False,
                                   learning_rate_placeholder: 0.0
                               })
        emb_array[lab, :] = emb

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gpu_memory_fraction", type=float, help="Upper bound on the amount of GPU memory that will be used by the process.", default=0.3)
    parser.add_argument('--max_nrof_epochs', type=int, help="Number of epochs to run", default=500)
    parser.add_argument("--logs_base_dir", type=str, help='Directory where to write event logs.', default='logs/')
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
