# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from scipy import misc

import align.detect_face as detect_face


def main(args):
    raw_data_path = os.path.expanduser(args.raw_data_path)
    aligned_data_path = os.path.expanduser(args.output_path)

    image_paths, nrof_classes = load_image_paths(raw_data_path)



    align_face_and_output(
        image_paths=image_paths,
        minsize=args.minsize,
        threshold=args.threshold,
        factor=args.factor,
        margin=args.margin,
        multi_detect=args.multi_detect,
        face_size=args.face_size
    )

def align_face_and_output(image_paths, minsize, threshold, factor, margin, multi_detect, face_size):
    class_labels = np.arange(len(image_paths))
    with tf.Graph().as_default():
        with tf.Session() as session:
            pnet, rnet, onet = detect_face.create_mtcnn(session, None)
            for class_label in class_labels:
                same_person_image_paths = image_paths[class_label]
                # print(class_label, same_person_image_paths)
                for serial_id, image_path in enumerate(same_person_image_paths):
                    image = misc.imread(image_path, mode="RGB")
                    img_size = np.asarray(image.shape)[0: 2]
                    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet,
                                                                threshold, factor)
                    if not multi_detect:    # just detect one face
                        bounding_box = np.squeeze(bounding_boxes[0])
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(bounding_box[0] - margin / 2, 0)  # x1
                        bb[1] = np.maximum(bounding_box[1] - margin / 2, 0)  # y1
                        bb[2] = np.minimum(bounding_box[2] + margin / 2, img_size[1])  # x2
                        bb[3] = np.minimum(bounding_box[3] + margin / 2, img_size[0])  # y2

                        cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
                        aligned = misc.imresize(cropped, (face_size, face_size), interp="bilinear")
                        # unique_id = str(class_label) + "_"
                        # output_filename = "{}/{}_{}{}".format(unique_id, unique_id, )

                    else:
                        pass

def load_image_paths(raw_data_path):
    nrof_classes = []
    all_image_paths = []
    for root, dirs, files in os.walk(raw_data_path):
        # if it has no dirs in this root, this directory is the same person directory
        if len(dirs) > 0:
            continue
        image_paths = [os.path.join(root, file) for file in files]
        nrof_same_person = len(image_paths)
        nrof_classes.append(nrof_same_person)
        all_image_paths.append(image_paths)

    return all_image_paths, nrof_classes

def sample(image_paths, nrof_classes):
    pass


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_path", type=str, help="The path of raw image data")
    parser.add_argument("output_path", type=str, help="The path of output")
    parser.add_argument("--minsize", type=int, help="minimum size of face", default=20)
    parser.add_argument("--threshold", type=list, help="three model threshold", default=[0.6, 0.7, 0.7])
    parser.add_argument("--factor", type=float, help="scale factor", default=0.709)
    parser.add_argument("--face_size", type=int, help="the size of each cropped faces", default=224)
    parser.add_argument("--margin", type=int, help="Margin for the crop around the bounding box (height, width) "
                                                   "in pixels." , default=44)
    parser.add_argument("--file_ext", type=str, help="The extension of output image files", default="png")
    parser.add_argument("--multi_detect", type=bool, help="Is need the detect multi face in one image", default=False)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

