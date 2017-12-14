# -*- coding: utf-8 -*-

import sys
import os
import argparse

import numpy as np

import Nemo.valid.face as face_module

from sklearn import metrics

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
    def dist(self):
        if np.isnan(self._dist):
            self._calculate_dist()
        return self._dist

    def _calculate_dist(self):
        diff = np.subtract(self.embedding, self._family_domain_vector)
        self._dist = np.sum(np.square(diff))

def main(args):
    root_dir = os.path.expanduser(args.data_dir)
    domain_dir = os.path.join(root_dir, "domain")
    margin = args.margin

    is_domains = []
    domain_vectors = []
    img_infos = []
    for family_id in os.listdir(domain_dir):
        family_dir = os.path.join(domain_dir, family_id)
        nrof_family_images = 0
        domain_vector = np.zeros(128)
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
                    img_infos.append((image_path, family_id))
                    is_domains.append(isdomain)
                    nrof_family_images += 1
            elif _path.endswith(".npy"):
                domain_vector = np.load(_path)
        domain_vectors += [domain_vector] * nrof_family_images

    assert len(is_domains) == len(domain_vectors)
    assert len(is_domains) == len(img_infos)

    faces = face_module.detect_faces(img_infos, args.minsize, args.mtcnn_threshold, args.factor, margin, args.face_size)
    # convert
    sample_faces = [SampleFace(face, domain_vector, is_domain)
                    for face, domain_vector, is_domain in zip(faces, domain_vectors, is_domains)]
    is_domains = np.asarray(is_domains)

    face_module.calculate_embeddings(sample_faces, args.facenet_model_dir, args.batch_size)

    thresholds = np.arange(0, 2, 0.001)
    tprs = []
    fprs = []
    accuracies = []
    for threshold in thresholds:
        tpr, fpr, accuracy = predict_is_domain(sample_faces, is_domains, threshold)
        tprs.append(tpr)
        fprs.append(fpr)
        accuracies.append(accuracy)

    auc = metrics.auc(fprs, tprs)
    print("Accuracy: %1.3f+-%1.3f" % (np.mean(accuracies), np.std(accuracies)))
    print('Area Under Curve (AUC): %1.3f' % auc)
    # print(np.mean(tprs), np.mean(fprs), np.mean(accuracies))

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
    parser.add_argument("--mtcnn_threshold", type=list, help="three model threshold", default=[0.6, 0.7, 0.7])
    parser.add_argument("--factor", type=float, help="scale factor", default=0.709)
    parser.add_argument("--margin", type=int, help="Margin for the crop around the bounding box (height, width) "
                                                   "in pixels.", default=44)
    parser.add_argument("--batch_size", type=int, help="the enqueue batch size", default=3)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

