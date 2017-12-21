# -*- coding: utf-8 -*-

import argparse
import sys
import os
import re
from shutil import copyfile

import tensorflow as tf
from tensorflow.python.platform import gfile

import numpy as np
import sklearn

from scipy import misc
from sklearn.mixture import GMM
from sklearn.cluster import KMeans

def main(args):
    root_dir = os.path.expanduser(args.data_dir)
    model_dir = os.path.expanduser(args.model_dir)
    pp_root_dir = os.path.join(root_dir, "preprocess")
    p_root_dir = os.path.join(root_dir, "process")
    save_dir = os.path.join(root_dir, "save")

    if not os.path.isdir(p_root_dir):
        os.mkdir(p_root_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with tf.Session() as session:
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())

        load_model(model_dir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")


        for family_id in os.listdir(pp_root_dir):
            preprocess_family_dir = os.path.join(pp_root_dir, family_id)
            process_family_dir = os.path.join(p_root_dir, family_id)
            save_family_dir = os.path.join(save_dir, family_id)
            if not os.path.exists(save_family_dir):
                os.makedirs(save_family_dir)
            family_image_paths = []

            if not os.path.isdir(process_family_dir):
                os.mkdir(process_family_dir)

            for (root, dirs, files) in os.walk(preprocess_family_dir):
                if len(dirs) == 0:
                    family_image_paths += [os.path.join(root, file) for file in files]

            nrof_images = len(family_image_paths)

            idx_array = np.arange(nrof_images)  # 记录每张脸的index
            emb_array = np.zeros((nrof_images, 128))  # 没张脸的向量信息
            nrof_batch = int(np.ceil(nrof_images / args.batch_size))
            for label in range(nrof_batch):
                start_index = label * args.batch_size
                end_index = min((label + 1) * args.batch_size, nrof_images)
                images = load_data(family_image_paths[start_index: end_index], args.image_size)
                emb_array[start_index: end_index, :] = session.run(embeddings, feed_dict={images_placeholder: images, phase_train_placeholder: False})

            print("finish calculation of current %s-th family face vectors" % family_id)

            # maybe use the unsupervised learning to category face.
            # TODO: Here, I use greedy algorithm, but I think this is a good strategy
            face_array = greed_cluster(emb_array, idx_array, args.threshold)

            print("clustering the face and there are %d categories" % len(face_array))

            process_cluster(face_array, process_family_dir, emb_array, save_family_dir, family_image_paths, root_dir, family_id)
            print("----------------------------------------\n")

def process_cluster(face_array, process_family_dir, emb_array, save_family_dir,
                    family_image_paths, root_dir, family_id):
    representations = []
    largest_area = 0
    domain_image_path = ""
    domain_represent_vector = np.zeros(128)

    # copy file to the process dir
    for label, face_ids in enumerate(face_array):
        label_dir = os.path.join(process_family_dir, str(label))
        os.mkdir(label_dir)

        represent_embedding, represent_id = near_center(emb_array, face_ids)
        representations.append(represent_embedding)
        np.save(os.path.join(save_family_dir, str(label)), represent_embedding)

        current_area = float(family_image_paths[represent_id].split("_")[-3])
        if current_area > largest_area:
            largest_area = current_area
            domain_id = represent_id
            domain_image_path = family_image_paths[domain_id]
            domain_represent_vector = represent_embedding
            domain_label = label

        for j, face_index in enumerate(face_ids):
            image_name = family_image_paths[face_index].split("/")[-1]
            copyfile(family_image_paths[face_index], os.path.join(label_dir, image_name))

    save_domain_info(root_dir, domain_image_path, family_id, domain_represent_vector, domain_label)
    print("save the domain information, current family id : %s, domain_label : %s" %
          (str(family_id), str(domain_label)))

def save_domain_info(root_dir, image_path, family_id, represent_vector, label):
    domain_root_dir = os.path.join(root_dir, "domain")
    if not os.path.exists(domain_root_dir):
        os.makedirs(domain_root_dir)

    family_dir = os.path.join(domain_root_dir, family_id)
    if not os.path.exists(family_dir):
        os.makedirs(family_dir)

    np.save(os.path.join(family_dir, str(label)), represent_vector)

    image_name = image_path.split("/")[-1]
    copyfile(image_path, os.path.join(family_dir, image_name))

def calculate_cluster(strategy):
    pass

def greed_cluster(embeddings, idx_array, threshold):
    face_array = []  # 保存分类信息
    while len(idx_array) > 0:
        index = idx_array[0]
        base_emb = embeddings[index]
        dist = np.sum(np.square(np.subtract(base_emb, embeddings[idx_array, :])), 1)
        idx = np.where(dist < threshold)[0].tolist()
        face_array.append(idx_array[idx])
        idx_array = np.delete(idx_array, idx)
    return face_array

def near_center(embeddings, face_ids):
    """
    which embedding is near the center of cluster and the id of this embedding
    :param embeddings: All the embeddings in family
    :param face_ids: the id of current cluster
    :return:
    """
    cluster_embeddings = embeddings[face_ids]
    center = np.mean(cluster_embeddings)
    diff = np.subtract(cluster_embeddings, center)
    dist = np.sum(np.square(diff), 1)
    main_id = np.argmin(dist)
    return cluster_embeddings[main_id], face_ids[main_id]

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
        images[i] = prewhiten(images[i])
    return images

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

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
