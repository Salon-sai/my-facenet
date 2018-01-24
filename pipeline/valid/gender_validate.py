# -*- coding: utf-8 -*-

import argparse
import os
import sys
import shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import misc
from data_process import utils
from pipeline.process import process
from pipeline.valid.face import Face

class GenderFace(Face):

    def __init__(self, aligned_image, family_id, image_path, actual_gender):
        Face.__init__(self, aligned_image, family_id, image_path)
        self._actual_gender = actual_gender
        self._predict_gender = -1

    @property
    def predict_gender(self):
        return self._predict_gender

    @property
    def actual_gender(self):
        return self._actual_gender

    @predict_gender.setter
    def predict_gender(self, predict_gender):
        self._predict_gender = predict_gender


def main(args):
    test_aligned_path = os.path.expanduser(args.test_dir)

    if not os.path.isdir(test_aligned_path):
        os.mkdir(test_aligned_path)

    img_infos = []
    for root, dirs, files in os.walk(test_aligned_path):
        if len(dirs) == 0:
            family_id = root.split("/")[-2]
            gender = int(root.split("_")[-1])
            # gender = root.split("_")[-1]
            img_infos.append([[os.path.join(root, file), family_id, gender] for file in files])

    with tf.Session() as session:
        session.run(tf.local_variables_initializer())
        session.run(tf.global_variables_initializer())

        process.load_model(args.facenet_model_dir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        faces = []
        if not args.need_align:
            for same_people_images_path in img_infos:
                same_faces = []
                for image_path, family_id, gender in same_people_images_path:
                    image = misc.imread(image_path)
                    assert image.shape == (args.image_size, args.image_size, 3)
                    image = utils.prewhiten(image)
                    same_faces.append(GenderFace(image, family_id, image_path, gender))
                images = [face.aligned_image for face in same_faces]
                emb_array = session.run(embeddings, feed_dict={
                    images_placeholder: images,
                    phase_train_placeholder: False
                })

                for i, emb in enumerate(emb_array):
                    same_faces[i].embedding = emb
                faces.append(same_faces)
        else:
            print("Please aligned the images")

    with tf.Session() as session:

        process.load_model(args.gender_model_dir)

        embeddings_placeholder = tf.get_default_graph().get_tensor_by_name("embeddings_placeholder:0")

        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_gender_train:0")

        predict = tf.get_default_graph().get_tensor_by_name("predict:0")

        for same_faces in faces:
            emb_array = [face.embedding for face in same_faces]
            preds = session.run(predict, feed_dict={
                embeddings_placeholder: emb_array,
                phase_train_placeholder: False
            })

            for i, face in enumerate(same_faces):
                face.predict_gender = preds[i]

    calculate_roc(faces)

def calculate_roc(faces):
    thresholds = np.arange(0, 1, 0.005)
    num_thresholds = len(thresholds)

    accuracy_array = np.empty(num_thresholds)
    num_people = len(faces)

    correct_rate = np.empty((len(faces)))
    all_correct = 0
    num_all_faces = 0

    root_dir = os.path.expanduser("~/data/label_task_v1/gender_incorrect")
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    for i, same_faces in enumerate(faces):
        num_faces = len(same_faces)
        predict_genders = np.zeros((len(same_faces)), dtype=np.int32)
        actual_genders = np.zeros((len(same_faces)), dtype=np.int32)
        for ii, face in enumerate(same_faces):
            predict_genders[ii] = face.predict_gender
            actual_genders[ii] = face.actual_gender

            if face.predict_gender != face.actual_gender:
                prefix_image = face.image_path.split("/")[-3:]
                save_dir = os.path.join(root_dir, prefix_image[0], prefix_image[1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                shutil.copy(face.image_path, save_dir)

        num_correct = np.sum(np.equal(predict_genders, actual_genders))
        correct_rate[i] = float(num_correct) / num_faces

        num_all_faces += num_faces
        all_correct += num_correct

    accuracy = all_correct / num_all_faces

    for i, threshold in enumerate(thresholds):
        accuracy_array[i] = float(np.sum(correct_rate > threshold)) / num_people

    plt.plot(thresholds, accuracy_array)
    plt.ylabel("Accuracy Rate")
    plt.xlabel("Thresholds")
    # plt.show()

    print('Accuracy: %1.4f' % accuracy)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", type=str, help="The directory of test aligned data")
    parser.add_argument("facenet_model_dir", type=str, help="The face net model of directory")
    parser.add_argument("gender_model_dir", type=str, help="The directory of gender model")
    parser.add_argument("--batch_size", type=int, help="the enqueue batch size", default=3)
    parser.add_argument("--need_align", type=bool, help="Whether face images need to align", default=False)
    parser.add_argument("--image_size", type=int, help="The size of face image", default=160)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
