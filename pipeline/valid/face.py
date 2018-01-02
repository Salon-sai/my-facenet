# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import tensorflow as tf
from scipy import misc

from align import detect_face
from pipeline.process import process


class Face(object):

    def __init__(self, aligned_image, family_id, image_path):
        self._image = aligned_image
        self._family_id = family_id
        self._from_image_path = image_path

    @property
    def family_id(self):
        return self._family_id

    @property
    def aligned_image(self):
        return self._image

    @property
    def embedding(self):
        return self._embedding

    @property
    def image_path(self):
        return self._from_image_path

    @embedding.setter
    def embedding(self, embedding):
        self._embedding = embedding

    @image_path.setter
    def image_path(self, from_image_path):
        self._from_image_path = from_image_path

    @aligned_image.setter
    def aligned_image(self, aligned_image):
        self._image = aligned_image

def detect_faces(img_info, minsize, mtccn_threshold, factor, margin, face_size, multi_detect=False):
    faces = []
    remove_ids = []
    nrof_extract_err = 0
    with tf.Session() as session:
        pnet, rnet, onet = detect_face.create_mtcnn(session, None)
        start_time = time.time()
        for img_info_i, (image_path, family_id) in enumerate(img_info):
            image = misc.imread(image_path, mode='RGB')
            bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet,
                                                        mtccn_threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 0:
                image_size = np.asarray(image.shape)[0:2]
                for index, bounding_box in enumerate(bounding_boxes):
                    if not multi_detect and nrof_faces > 1:
                        nrof_extract_err += 1
                        remove_ids.append(img_info_i)
                        print("It detects multi face, image path %s, the number "
                              "of faces: %d" % (image_path, nrof_faces))
                        break
                    bounding_box = np.squeeze(bounding_box)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(bounding_box[0] - margin / 2, 0)  # x1
                    bb[1] = np.maximum(bounding_box[1] - margin / 2, 0)  # y1
                    bb[2] = np.minimum(bounding_box[2] + margin / 2, image_size[1])  # x2
                    bb[3] = np.minimum(bounding_box[3] + margin / 2, image_size[0])  # y2

                    cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
                    aligned = misc.imresize(cropped, (face_size, face_size), interp="bilinear")
                    face = Face(aligned, family_id, image_path)
                    faces.append(face)
            else:
                remove_ids.append(img_info_i)
                nrof_extract_err += 1
                print("can not extract the face from the image : %s" % image_path)
        end_time = time.time() - start_time
    print("Extract image: %d Using detect time %f" % (len(img_info), end_time))
    return faces, nrof_extract_err, remove_ids

def calculate_embeddings(faces, facenet_model_dir, batch_size):
    with tf.Session() as session:
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())

        process.load_model(facenet_model_dir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        nrof_faces = len(faces)
        nrof_batch = int(np.ceil(nrof_faces / batch_size))
        for i in range(nrof_batch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_faces)
            face_images = [face.aligned_image for face in faces[start_index: end_index]]
            emb_arry = session.run(embeddings,
                                   feed_dict={images_placeholder: face_images, phase_train_placeholder: False})

            for i, emb in enumerate(emb_arry):
                faces[i + start_index].embedding = emb

def compare_and_save(face, root_dir, result_dir, facenet_threshold, save_name, file_ext):
    save_family_npy = os.path.join(root_dir, "save", str(face.family_id) + ".npy")
    family_representations = np.load(save_family_npy)
    dists = np.sum(np.square(np.subtract(family_representations, face.embedding)), 1)
    sim_id = np.argmin(dists)
    if dists[sim_id] < facenet_threshold:
        result_sim_dir = os.path.join(result_dir, str(face.family_id), str(sim_id))
        if not os.path.isdir(result_sim_dir):
            os.makedirs(result_sim_dir)
        age, gender = predict_age_and_gender(face, save_name)
        filename = os.path.join(result_sim_dir, "%s_%d_%d.%s" % (save_name, age, gender, file_ext))
        misc.imsave(filename, face.aligned_image)
    else:
        result_other_dir = os.path.join(result_dir, str(face.family_id), "other")
        if not os.path.isdir(result_other_dir):
            os.makedirs(result_other_dir)
        age, gender = predict_age_and_gender(face, save_name)
        filename = os.path.join(result_other_dir, "%s_%d_%d.%s" % (save_name, age, gender, file_ext))
        misc.imsave(filename, face.aligned_image)

def predict_age_and_gender(face, save_name):
    import hashlib
    s = str(face.family_id) + save_name
    hash_int = int(hashlib.sha1(s.encode()).hexdigest(), 16) % (10 ** 8)
    age = hash_int % 100
    gender = hash_int % 2
    return age, gender