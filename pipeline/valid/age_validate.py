# -*- coding: utf-8 -*-

import sys
import os
import argparse

import tensorflow as tf
import numpy as np

from scipy import misc
from data_process import utils
from pipeline.process import process
from pipeline.valid.face import Face

class AgeFace(Face):

    def __init__(self, aligned_image, family_id, image_path, actual_age):
        Face.__init__(self, aligned_image, family_id, image_path)
        self._actual_age = actual_age
        self._predict_age = -1

    @property
    def predict_age(self):
        return self._predict_age

    @property
    def actual_age(self):
        return self._actual_age

    @predict_age.setter
    def predict_age(self, predict_age):
        self._predict_age = predict_age

def main(args):
    test_aligned_path = os.path.expanduser(args.test_dir)

    if not os.path.isdir(test_aligned_path):
        os.mkdir(test_aligned_path)

    img_infos = []
    num_samples = 0
    for root, dirs, files in os.walk(test_aligned_path):
        if len(dirs) == 0:
            family_id = root.split("/")[-2]
            age = int(root.split("_")[-1])
            # gender = root.split("_")[-1]
            img_infos.append([[os.path.join(root, file), family_id, age] for file in files])

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
                    same_faces.append(AgeFace(image, family_id, image_path, gender))
                    num_samples += 1
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

    correct_sum = 0
    with tf.Session() as session:

        process.load_model(args.age_model_dir)

        embeddings_placeholder = tf.get_default_graph().get_tensor_by_name("embeddings_placeholder:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_age_train:0")
        predict = tf.get_default_graph().get_tensor_by_name("predict:0")

        for same_faces in faces:
            emb_array = [face.embedding for face in same_faces]
            preds = session.run(predict, feed_dict={
                embeddings_placeholder: emb_array,
                phase_train_placeholder: False
            })

            for i, face in enumerate(same_faces):
                face.predict_age = preds[i]
                if face.actual_age == face.predict_age:
                    correct_sum += 1

    print("Accuracy: %1.4f" % float(correct_sum / num_samples))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", type=str, help="The directory of test aligned data")
    parser.add_argument("facenet_model_dir", type=str, help="The face net model of directory")
    parser.add_argument("age_model_dir", type=str, help="The directory of gender model")
    parser.add_argument("--batch_size", type=int, help="the enqueue batch size", default=3)
    parser.add_argument("--need_align", type=bool, help="Whether face images need to align", default=False)
    parser.add_argument("--image_size", type=int, help="The size of face image", default=160)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
