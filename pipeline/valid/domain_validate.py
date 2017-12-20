# -*- coding: utf-8 -*-

import sys
import os
import argparse

import numpy as np
import tensorflow as tf

from shutil import copyfile
from scipy import misc

import pipeline.valid.face as face_module
import matplotlib.pyplot as plt

from pipeline.preprocess.align import detect_face

from sklearn import metrics
from sklearn.preprocessing import normalize, MinMaxScaler

class SampleFace(face_module.Face):
    def __init__(self, face, family_domain_vector, is_domain):
        face_module.Face.__init__(self, face.aligned_image, face.family_id, face.image_path)
        self._family_domain_vector = family_domain_vector
        self._is_domain = is_domain
        self._predict = False
        self._dist = np.NAN

    def predict(self, threshold):
        self._predict = threshold > self.dist
        return self._predict

    @property
    def is_domain(self):
        return self._is_domain

    @property
    def dist(self):
        if np.isnan(self._dist):
            self._calculate_dist()
        return self._dist

    def _calculate_dist(self):
        diff = np.subtract(self.embedding, self._family_domain_vector)
        self._dist = np.sum(np.square(diff))

    def max_min(self, vector):
        max_value = np.max(vector)
        min_value = np.min(vector)
        return (vector - min_value) / (max_value - min_value)

def main(args):
    root_dir = os.path.expanduser(args.data_dir)
    domain_dir = os.path.join(root_dir, "domain")
    margin = args.margin

    sample_faces = []
    with tf.Session() as session:
        pnet, rnet, onet = detect_face.create_mtcnn(session, None)
        for family_id in os.listdir(domain_dir):
            family_dir = os.path.join(domain_dir, family_id)
            nrof_family_images = 0
            # first we need the domain vector
            domain_vector = [np.load(os.path.join(family_dir, filename)) for filename in os.listdir(family_dir) if filename.endswith(".npy")][0]
            for filename in os.listdir(family_dir):
                _path = os.path.join(family_dir, filename)
                if os.path.isdir(_path):
                    sample_dir = _path
                    if filename == "true":
                        isdomain = True
                    else:
                        isdomain = False
                    for image_name in os.listdir(sample_dir):
                        image_path = os.path.join(sample_dir, image_name)

                        image = misc.imread(image_path, mode='RGB')
                        bounding_boxes, _ = detect_face.detect_face(image, args.minsize, pnet, rnet, onet,
                                                                    args.mtcnn_threshold, args.factor)
                        image_size = np.asarray(image.shape)[0:2]

                        if not args.multi_detect:
                            if len(bounding_boxes) == 1:
                                bounding_box = bounding_boxes[0]
                                bounding_box = np.squeeze(bounding_box)
                                bb = np.zeros(4, dtype=np.int32)
                                bb[0] = np.maximum(bounding_box[0] - margin / 2, 0)  # x1
                                bb[1] = np.maximum(bounding_box[1] - margin / 2, 0)  # y1
                                bb[2] = np.minimum(bounding_box[2] + margin / 2, image_size[1])  # x2
                                bb[3] = np.minimum(bounding_box[3] + margin / 2, image_size[0])  # y2

                                cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
                                aligned = misc.imresize(cropped, (args.face_size, args.face_size), interp="bilinear")
                                aligned = prewhiten(aligned)
                                face = face_module.Face(aligned, family_id, image_path)
                                sample_face = SampleFace(face, domain_vector, isdomain)
                                sample_faces.append(sample_face)
                                nrof_family_images += 1

    face_module.calculate_embeddings(sample_faces, args.facenet_model_dir, args.batch_size)

    is_domains = np.asarray([sample_face.is_domain for sample_face in sample_faces])

    thresholds = np.arange(0, 2, 0.01)
    tprs = []
    fprs = []
    accuracies = []
    for threshold in thresholds:
        tpr, fpr, accuracy = predict_is_domain(sample_faces, is_domains, threshold)
        tprs.append(tpr)
        fprs.append(fpr)
        accuracies.append(accuracy)

    auc = metrics.auc(fprs, tprs)
    plt.plot(fprs, tprs, color="r", label="ROC")
    plt.plot([0, 1], [0, 1], '--',color="b", label="Base")
    plt.ylabel("True Positive")
    plt.xlabel("False Positive")


    print("Best Accuracy: %1.3f, When threshold: %1.4f, ture positive: %1.3f, false positive: %1.3f" %
          (np.max(accuracies), thresholds[np.argmax(accuracies)],
                                          tprs[int(np.argmax(accuracies))], fprs[int(np.argmax(accuracies))]))
    print("Accuracy: %1.3f+-%1.3f" % (np.mean(accuracies), np.std(accuracies)))
    print("Total number of positive sample: %d, Total number of negative sample: %d"
          % (np.sum(is_domains), np.sum(np.logical_not(is_domains))))
    print("True Positive rate: %1.3f+-%1.3f, False Positive rate: %1.3f+-%1.3f" %
          (np.mean(tprs), np.std(tprs), np.mean(fprs), np.std(fprs)))
    print('Area Under Curve (AUC): %1.3f' % auc)
    plt.legend()
    plt.title('Area Under Curve (AUC): %1.3f' % auc)
    plt.show()
    # print(np.mean(tprs), np.mean(fprs), np.mean(accuracies))

def category_family(faces):
    categoried = dict()
    for face in faces:
        categoried.keys()

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def predict_is_domain(sample_faces, actual_is_domain, threshold):
    predicts = [sample_face.predict(threshold) for sample_face in sample_faces]

    tp = np.sum(np.logical_and(actual_is_domain, predicts))
    fp = np.sum(np.logical_and(np.logical_not(actual_is_domain), predicts))
    tn = np.sum(np.logical_and(np.logical_not(actual_is_domain), np.logical_not(predicts)))
    fn = np.sum(np.logical_and(actual_is_domain, np.logical_not(predicts)))

    tpr = 0 if (tp+fn == 0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)

    acc = float(tp + tn) / actual_is_domain.size

    return tpr, fpr, acc

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="The directory of data set.")
    parser.add_argument("facenet_model_dir", type=str, help="The directory of FaceNet model")
    parser.add_argument("--face_size", type=int, help="The cropped size of face", default=160)
    parser.add_argument("--minsize", type=int, help="minimum size of face", default=20)
    parser.add_argument("--mtcnn_threshold", type=list, help="three model threshold", default=[0.8, 0.8, 0.8])
    parser.add_argument("--factor", type=float, help="scale factor", default=0.709)
    parser.add_argument("--margin", type=int, help="Margin for the crop around the bounding box (height, width) "
                                                   "in pixels.", default=44)
    parser.add_argument("--batch_size", type=int, help="the enqueue batch size", default=3)
    parser.add_argument("--multi_detect", type=bool, help="Multi detect the face", default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

