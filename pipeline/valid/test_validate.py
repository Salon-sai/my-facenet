# -*- coding: utf-8 -*-

import argparse
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy import misc
from pipeline.preprocess import utils
from pipeline.valid.face import detect_faces, Face, calculate_embeddings

class LabelFace(Face):

    def __init__(self, face, label, save_family_dir):
        Face.__init__(self, face.aligned_image, face.family_id, face.image_path)
        self._label = label
        self._save_family_dir = save_family_dir

    @property
    def label(self):
        try:
            return int(self._label)
        except ValueError:
            if isinstance(self._label, str):
                return -1
            else:
                raise Exception("the label must be number or string, but got %s " % type(self._label))

    @property
    def save_family_dir(self):
        return self._save_family_dir

def main(args):
    test_path = os.path.expanduser(args.test_dir)
    save_path = os.path.expanduser(args.save_dir)
    test_align_path = test_path + "_align"
    if not os.path.isdir(test_align_path):
        os.mkdir(test_align_path)

    # step1: align images and save them
    img_infos = []
    for root, dirs, files in os.walk(test_path):
        if len(dirs) == 0:
            family_id = root.split("/")[-2]
            img_infos += [[os.path.join(root, file), family_id] for file in files]

    img_infos = np.asarray(img_infos)
    if args.need_align:
        faces, _, _ = detect_faces(img_infos, args.minsize, args.mtcnn_threshold,
                                            args.factor, args.margin, args.image_size)
    else:
        faces = []
        for img_path, family_id in img_infos:
            faces.append(Face(misc.imread(img_path), family_id, img_path))

    label_faces = []
    for face in faces:
        base_name = face.image_path.split("/")[-1]
        label = face.image_path.split("/")[-2]
        align_dir_path = os.path.join(test_align_path, face.family_id, label)
        current_save_family = os.path.join(save_path, face.family_id)
        if not os.path.exists(align_dir_path):
            os.makedirs(align_dir_path)
        align_image_path = os.path.join(align_dir_path, base_name)
        misc.imsave(align_image_path, face.aligned_image)
        # convert Face to Face with label
        label_face = LabelFace(face, label, current_save_family)
        label_face.image_path = align_image_path
        label_face.aligned_image = utils.prewhiten(label_face.aligned_image)
        label_faces.append(label_face)

    # step2: use align face image to predict which person is
    current_family_id = -1
    nrof_faces = len(label_faces)
    represent_embeddings = []
    calculate_embeddings(label_faces, args.facenet_model_dir, args.batch_size)
    thresholds = np.arange(0, args.max_threshold, 0.1)
    accuracies = np.zeros((len(thresholds)))    # whether the person in this family
    classify_acc = np.zeros((len(thresholds)))

    for index, threshold in enumerate(thresholds):
        tp = 0; fp = 0; tn = 0; fn = 0
        fail_classify_in_family = 0
        success_classify_in_family = 0
        for label_face in label_faces:
            if not current_family_id == label_face.family_id:
                current_family_id = label_face.family_id
                labels = os.listdir(label_face.save_family_dir)
                represent_embeddings = np.empty(shape=(len(labels), 128), dtype=np.float32)
                for label in labels:
                    embedding_file = os.path.join(label_face.save_family_dir, label, "embedding.npy")
                    represent_embeddings[int(label), :] = np.load(embedding_file)
                represent_embeddings = np.asarray(represent_embeddings)
            diff = np.subtract(represent_embeddings, label_face.embedding)
            dist = np.sum(np.square(diff), 1)
            mini_dist = np.min(dist)
            if mini_dist > threshold:
                # predict as negative (not in current family)
                tn += label_face.label == -1
                if label_face.label > -1:
                    fn += 1
                    # classify person in current family was fail
                    fail_classify_in_family += 1
            else:
                # predict as positive (in current family)
                fp += label_face.label == -1
                if label_face.label > -1:
                    tp += 1
                    predict_label = np.argmin(dist)
                    # classify person in current family
                    success_classify_in_family += predict_label == label_face.label
                    fail_classify_in_family += predict_label != label_face.label

        accuracies[index] = float(tp + tn) / float(nrof_faces)
        classify_acc[index] = float(success_classify_in_family) / float(success_classify_in_family +
                                                                        fail_classify_in_family)

    print("Classify in same family or not, Accuracy: %1.3f+-%1.3f" % (accuracies.mean(), accuracies.std()))
    print("Best Accuracy of classification in same family or not %1.3f" % accuracies.max())
    print("Classify family members in same family %1.3f+-%1.3f" % (classify_acc.mean(), classify_acc.std()))
    print("Best Accuracy of classification family members in same family %1.3f" % classify_acc.max())
    print("Number of faces: %d, Number of classify images %d, Number of other images in all family %d " %
          (nrof_faces, (success_classify_in_family + fail_classify_in_family),
           nrof_faces - (success_classify_in_family + fail_classify_in_family)))

    plt.plot(thresholds, accuracies, color="r", label="Classify in same family or not")
    plt.plot(thresholds, classify_acc, '--', color="b", label="Classify family members in same family")

    plt.ylabel("Accuracy")
    plt.xlabel("Thresholds")

    plt.legend()
    plt.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", type=str, help="The directory of test")
    parser.add_argument("save_dir", type=str, help="The directory of saving each label embeddings and face image")
    parser.add_argument("facenet_model_dir", type=str, help="The face net model of directory")
    parser.add_argument("--image_size", type=int, help="The image size", default=160)
    parser.add_argument("--minsize", type=int, help="minimum size of face", default=100)
    parser.add_argument("--mtcnn_threshold", type=list, help="three model threshold", default=[0.6, 0.7, 0.7])
    parser.add_argument("--factor", type=float, help="scale factor", default=0.709)
    parser.add_argument("--margin", type=int, help="Margin for the crop around the bounding box (height, width) "
                                                   "in pixels.", default=44)
    parser.add_argument("--batch_size", type=int, help="the enqueue batch size", default=3)
    parser.add_argument("--max_threshold", type=float, help="The max threshold of threshold", default=2)
    parser.add_argument("--need_align", type=bool, help="Whether face images need to align", default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))