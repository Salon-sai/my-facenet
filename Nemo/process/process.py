# -*- coding: utf-8 -*-

import argparse
import sys
import os
import re
from shutil import copyfile

import tensorflow as tf
from tensorflow.python.platform import gfile

import numpy as np

from scipy import misc


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
            pp_family_dir = os.path.join(pp_root_dir, family_id)
            p_family_dir = os.path.join(p_root_dir, family_id)
            # s_family_npy = os.path.join(save_dir, family_id)
            s_family_dir = os.path.join(save_dir, family_id)
            if not os.path.exists(s_family_dir):
                os.makedirs(s_family_dir)
            family_image_paths = []

            if not os.path.isdir(p_family_dir):
                os.mkdir(p_family_dir)

            for (root, dirs, files) in os.walk(pp_family_dir):
                if len(dirs) == 0:
                    family_image_paths += [os.path.join(root, file) for file in files]

            nrof_images = len(family_image_paths)

            idx_array = np.arange(nrof_images)  # 记录每张脸的index
            emb_arry = np.zeros((nrof_images, 128))  # 没张脸的向量信息
            nrof_batch = int(np.ceil(nrof_images / args.batch_size))
            for label in range(nrof_batch):
                start_index = label * args.batch_size
                end_index = min((label + 1) * args.batch_size, nrof_images)
                images = load_data(family_image_paths[start_index: end_index], args.image_size)
                emb_arry[start_index: end_index, :] = session.run(embeddings, feed_dict={images_placeholder: images, phase_train_placeholder: False})

            print("finish calculation of current %s-th family face vectors" % family_id)

            # TODO: maybe use the unsupervised learning to category face. Here, I use greedy algorithm, but I think this is a good strategy
            face_array = []  # 保存分类信息
            while len(idx_array) > 0:
                index = idx_array[0]
                base_emb = emb_arry[index]
                dist = np.sum(np.square(np.subtract(base_emb, emb_arry[idx_array, :])), 1)
                idx = np.where(dist < args.threshold)[0].tolist()
                face_array.append(idx_array[idx])

                idx_array = np.delete(idx_array, idx)

            print("clustering the face and there are %d categories" % len(face_array))

            representations = []
            test_domain_id = -1
            test_domain_image_path = ""
            test_domain_represent_vector = np.zeros(128)
            test_domain_label = 0
            largest_area = 0

            # copy file to the process dir
            for label, face_ids in enumerate(face_array):
                label_dir = os.path.join(p_family_dir, str(label))
                os.mkdir(label_dir)

                for j, face_index in enumerate(face_ids):
                    if j == 0:
                        # In the clustering, we think the first face represents current cluster.
                        representations.append(emb_arry[face_index])
                        np.save(os.path.join(s_family_dir, str(label)), emb_arry[face_index])

                        current_area = float(family_image_paths[face_index].split("_")[-3])
                        if current_area > largest_area:
                            largest_area = current_area
                            test_domain_id = face_index
                            test_domain_image_path = family_image_paths[face_index]
                            test_domain_represent_vector = emb_arry[face_index]
                            test_domain_label = label

                    image_name = family_image_paths[face_index].split("/")[-1]
                    copyfile(family_image_paths[face_index], os.path.join(label_dir, image_name))

            domain_id, domain_image_path, domain_represent_vector, domain_label = find_domain_face(representations,
                                                                                     p_family_dir,
                                                                                     face_array,
                                                                                     family_image_paths)

            assert test_domain_id == domain_id
            assert test_domain_image_path == domain_image_path
            assert test_domain_represent_vector.all() == domain_represent_vector.all()
            assert test_domain_label == domain_label

            save_domain_info(root_dir, domain_image_path, family_id, domain_represent_vector, domain_label)
            print("save the domain information, current family id : %s, domain_label : %s" %
                  (str(family_id), str(domain_label)))

            print("----------------------------------------\n")
            # save the picture to directory of naming "save".
            # np.save(s_family_npy, np.asarray(representations))

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


def find_domain_face(represent_vectors, p_family_dir, face_array, family_image_paths):
    domain_id = -1
    domain_image_path = ""
    domain_represent_vector = np.zeros(128)
    domain_label = 0

    largest_area = 0
    for label, ids in enumerate(face_array):
        id = ids[0]
        label_dir = os.path.join(p_family_dir, str(label))
        image_path = family_image_paths[id]
        current_area = float(image_path.split("_")[-3])
        if current_area > largest_area:
            largest_area = current_area
            domain_id = id
            domain_image_path = image_path
            domain_represent_vector = represent_vectors[label]
            domain_label = label

    return domain_id, domain_image_path, domain_represent_vector, domain_label

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
    return images


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
