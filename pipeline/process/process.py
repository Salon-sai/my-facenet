# -*- coding: utf-8 -*-

import argparse
import sys
import os
import re
from shutil import copyfile

import tensorflow as tf
from tensorflow.python.platform import gfile
from scipy.stats import mode

import numpy as np

from scipy import misc
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score

def main(args):
    root_dir = os.path.expanduser(args.data_dir)
    identification_model_dir = os.path.expanduser(args.identification_model_dir)
    gender_model_dir = os.path.expanduser(args.gender_model_dir)
    age_model_dir = os.path.expanduser(args.age_model_dir)
    preprocess_root_dir = os.path.join(root_dir, "preprocess")
    process_root_dir = os.path.join(root_dir, args.process_subdir)
    save_dir = os.path.join(root_dir, args.save_subdir)

    if not os.path.isdir(process_root_dir):
        os.mkdir(process_root_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with tf.Session() as session:
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())

        load_model(identification_model_dir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        prelogits = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Bottleneck/BatchNorm/batchnorm/add_1:0")

        family_dict = dict()
        for family_id in os.listdir(preprocess_root_dir):
            preprocess_family_dir = os.path.join(preprocess_root_dir, family_id)
            process_family_dir = os.path.join(process_root_dir, family_id)
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

            # idx_array = np.arange(nrof_images)  # 记录每张脸的index
            emb_array = np.zeros((nrof_images, 128))  # embedding of each faces
            prelogits_array = np.zeros((nrof_images, 128))
            nrof_batch = int(np.ceil(nrof_images / args.batch_size))
            for label in range(nrof_batch):
                start_index = label * args.batch_size
                end_index = min((label + 1) * args.batch_size, nrof_images)
                images = load_data(family_image_paths[start_index: end_index], args.image_size)
                emb_array[start_index: end_index, :], prelogits_array[start_index: end_index] = \
                    session.run([embeddings, prelogits],
                                feed_dict={images_placeholder: images, phase_train_placeholder: False})

            print("finish calculation of current %s-th family face vectors" % family_id)

            print("clustering process...")
            # clusters_ids, represent_embeddings, represent_ids = kmean_cluster(emb_array, args.max_num_cluster)
            clusters_ids, represent_embeddings, represent_ids = greed_cluster(emb_array, args.threshold)
            print("clustering the face and there are %d categories" % len(clusters_ids))

            print("dominate process...")
            dominate_process(clusters_ids, represent_embeddings, represent_ids, family_image_paths, root_dir, family_id)

            save_process(represent_embeddings, represent_ids, family_image_paths,save_family_dir)

            person_emb_dict = process_cluster(clusters_ids, process_family_dir, emb_array, save_family_dir,
                            family_image_paths, root_dir, family_id)
            family_dict[family_id] = person_emb_dict
            print("----------------------------------------\n")

    with tf.Session(graph=tf.Graph()) as session:
        load_model(gender_model_dir, session)

        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings_placeholder:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_gender_train:0")
        predict = tf.get_default_graph().get_tensor_by_name("predict:0")

        for family_id, person_emb_dict in family_dict.items():
            new_person_emb_dict = dict()
            for person_id, person_emb_array in person_emb_dict.items():
                predict_genders = session.run(predict,
                                              feed_dict={embeddings: person_emb_array, phase_train_placeholder: False})
                predict_gender = mode(predict_genders, axis=None)[0][0]
                new_person_emb_dict[str(person_id) + "_" + str(predict_gender)] = person_emb_array
            family_dict[family_id] = new_person_emb_dict

    print("-------------------------\n")
    with tf.Session(graph=tf.Graph()) as session:
        load_model(age_model_dir, session)

        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings_placeholder:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_age_train:0")
        predict = tf.get_default_graph().get_tensor_by_name("predict:0")

        for family_id, person_emb_dict in family_dict.items():
            new_person_emb_dict = dict()
            for person_id, person_emb_array in person_emb_dict.items():
                predict_ages = session.run(predict,
                                           feed_dict={embeddings: person_emb_array, phase_train_placeholder: False})
                predict_age = mode(predict_ages, axis=None)[0][0]
                new_person_emb_dict[str(person_id) + "_" + str(predict_age)] = person_emb_array
            family_dict[family_id] = new_person_emb_dict

    save_txt = os.path.join(save_dir, "result.txt")
    with open(save_txt, "w") as f:
        for family_id, person_emb_dict in family_dict.items():
            for person_id_gender_age in person_emb_dict.keys():
                person_id, gender, age = person_id_gender_age.split("_")
                # if gender == '0':
                #     gender = 'F'
                # else:
                #     gender = 'M'
                #
                # if age == '0':
                #     age = 'child'
                # elif age == '1':
                #     age = 'adult'
                # elif age == '2':
                #     age = 'old'

                embedding_path = os.path.join(save_dir, str(family_id), person_id, "embedding.npy")

                f.write("%s,%s,%s,%s,%s\n" % (str(family_id), person_id, embedding_path, gender, age))


def process_cluster(face_array, process_family_dir, emb_array, save_family_dir,
                    family_image_paths, root_dir, family_id):
    """
    find the main face in current family and save the main face and this embedding vector
    :param face_array: clustered ids of face array
    :param process_family_dir: the name is "process" in current data directory
    :param emb_array: the embeddings of all family face
    :param save_family_dir: the name is "save" in current data directory
    :param family_image_paths: the images path of family in pre-process directory
    :param root_dir: the root directory of data
    :param family_id:
    :return:
    """
    representations = []
    largest_area = 0
    domain_image_path = ""
    domain_represent_vector = np.zeros(128)
    person_emb_dict = dict()

    # copy file to the process dir
    for label, face_ids in enumerate(face_array):
        label_dir = os.path.join(process_family_dir, str(label))
        os.mkdir(label_dir)

        # represent_embedding, represent_id = near_center(emb_array, face_ids)
        # representations.append(represent_embedding)
        # np.save(os.path.join(save_family_dir, str(label)), represent_embedding)

        # current_area = float(family_image_paths[represent_id].split("_")[-3])
        # # We guess the biggest face is main person in this family
        # # TODO: maybe the frequency of face is better measure for dominate face
        # if current_area > largest_area:
        #     largest_area = current_area
        #     domain_id = represent_id
        #     domain_image_path = family_image_paths[domain_id]
        #     domain_represent_vector = represent_embedding
        #     domain_label = label
        person_emb_dict[label] = emb_array[face_ids]
        for j, face_index in enumerate(face_ids):
            image_name = family_image_paths[face_index].split("/")[-1]
            copyfile(family_image_paths[face_index], os.path.join(label_dir, image_name))

    # save_domain_info(root_dir, domain_image_path, family_id, domain_represent_vector, domain_label)
    # print("save the domain information, current family id : %s, domain_label : %s" %
    #       (str(family_id), str(domain_label)))
    return person_emb_dict

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

def save_process(represent_embeddings, represent_ids, family_images_path, save_family_dir):
    """
    :param represent_embeddings: each embedding represents the corresponding cluster
    :param represent_ids:
    :param family_images_path:
    :param save_family_dir:
    :return:
    """
    for label, (represent_embedding, represent_id) in enumerate(zip(represent_embeddings, represent_ids)):
        label_path = os.path.join(save_family_dir, str(label))
        if not os.path.isdir(label_path):
            os.mkdir(label_path)
        np.save(os.path.join(label_path, "embedding"), represent_embedding)
        image_path = family_images_path[represent_id]
        image_name = image_path.split("/")[-1]
        copyfile(image_path, os.path.join(save_family_dir, str(label), image_name))

def dominate_process(clusters_ids, represent_embeddings, represent_ids, family_images_path, root_dir, family_id):
    """

    :param clusters_ids:
    :param represent_embeddings:
    :param represent_ids:
    :param family_images_path:
    :param root_dir:
    :param family_id:
    :return:
    """
    num_sample_cluster = [len(clusters_id) for clusters_id in clusters_ids]
    dominate_cluster_label = np.argmax(num_sample_cluster)
    dominate_embedding = represent_embeddings[dominate_cluster_label]
    dominate_image_id = represent_ids[dominate_cluster_label]
    dominate_image_path = family_images_path[dominate_image_id]
    save_domain_info(root_dir, dominate_image_path, family_id, dominate_embedding, dominate_cluster_label)

def kmean_cluster(embeddings, max_num_cluster):
    best_ch_score = -1
    best_kmean = None
    clusters_ids = []
    represent_embeddings = []
    represent_ids = []
    for num_cluster in range(2, max_num_cluster):
        kmean = KMeans(n_clusters=num_cluster).fit(embeddings)
        ch_score = calinski_harabaz_score(embeddings, kmean.labels_)
        if best_ch_score < ch_score or best_kmean is None:
            best_kmean = kmean
            best_ch_score = ch_score

    for label in np.unique(best_kmean.labels_):
        ids = np.where(best_kmean.labels_ == label)[0]
        represent_embedding, represent_id = near_center(embeddings, ids, center=best_kmean.cluster_centers_[label])

        clusters_ids.append(ids)
        represent_embeddings.append(represent_embedding)
        represent_ids.append(represent_id)

    return clusters_ids, represent_embeddings, represent_ids

def greed_cluster(embeddings, threshold):
    """
    using the greed algorithm to calculate the cluster of face
    :param embeddings: all embeddings of current family
    :param threshold: the discrimination of same face
    :return:
    """
    clusters_ids = []  # 保存分类信息
    represent_ids = []
    represent_embeddings = []
    idx_array = np.arange(embeddings.shape[0])
    while len(idx_array) > 0:
        index = idx_array[0]
        base_emb = embeddings[index]
        dist = np.sum(np.square(np.subtract(base_emb, embeddings[idx_array, :])), 1)
        idx = np.where(dist < threshold)[0].tolist()
        if len(idx) >= 2:
            clusters_ids.append(idx_array[idx])
        idx_array = np.delete(idx_array, idx)

    for cluster_ids in clusters_ids:
        represent_id = cluster_ids[0]
        represent_embedding = embeddings[represent_id]
        # represent_embedding, represent_id = near_center(embeddings, cluster_ids)

        represent_ids.append(represent_id)
        represent_embeddings.append(represent_embedding)

    return clusters_ids, represent_embeddings, represent_ids

def near_center(embeddings, face_ids, center=None):
    """
    which embedding is near the center of cluster and the id of this embedding
    :param embeddings: All the embeddings in family
    :param face_ids: the id of current cluster
    :return:
    """
    cluster_embeddings = embeddings[face_ids]
    if center == None:
        center = np.mean(cluster_embeddings)
    diff = np.subtract(cluster_embeddings, center)
    dist = np.sum(np.square(diff), 1)
    main_id = np.argmin(dist)
    return cluster_embeddings[main_id], face_ids[main_id]

def load_model(model, session=None):
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
        if session:
            saver.restore(session, os.path.join(model_path, ckpt_file))
        else:
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
    parser.add_argument("identification_model_dir", type=str, help="The directory of facenet cnn model")
    parser.add_argument("gender_model_dir", type=str, help="The directory of gender model")
    parser.add_argument("age_model_dir", type=str, help="The directory of age model")
    parser.add_argument("data_dir", type=str, help="the directory of data set")
    parser.add_argument("--process_subdir", type=str, help="The sub-directory of process", default="process")
    parser.add_argument("--save_subdir", type=str, help="The sub-directory of save", default="save")
    parser.add_argument("--image_size", type=int, help="the size of image when using the cnn", default=160)
    parser.add_argument("--batch_size", type=int, help="the enqueue batch size", default=3)
    parser.add_argument("--threshold", type=float, help="Use to classify the different and the same face", default=0.9)
    parser.add_argument("--max_num_cluster", type=int, help="each family has the max number of people "
                                                            "(number of clusters)", default=10)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
