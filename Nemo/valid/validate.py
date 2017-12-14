# -*- coding: utf-8 -*-

import argparse
import sys
import os

from Nemo.valid.face import detect_faces, calculate_embeddings, compare_and_save

def main(args):
    root_dir = os.path.expanduser(args.data_dir)
    validate_dir = os.path.join(root_dir, "validate")
    result_dir = os.path.join(root_dir, "result")
    margin = args.margin

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    img_info = []
    for family_id in os.listdir(validate_dir):
        family_dir = os.path.join(validate_dir, family_id)
        for file_name in os.listdir(family_dir):
            if os.path.isdir(file_name):
                continue
            image_path = os.path.join(family_dir, file_name)
            img_info.append((image_path, family_id))

    faces = detect_faces(img_info, args.minsize, args.mtccn_threshold, args.factor, margin, args.face_size)

    calculate_embeddings(faces, args.facenet_model_dir, args.batch_size)

    for i, face in enumerate(faces):
        compare_and_save(face, root_dir, result_dir, args.facenet_threshold, str(i), args.file_ext)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="The directory of data set.")
    parser.add_argument("facenet_model_dir", type=str, help="The directory of FaceNet model")
    parser.add_argument("--face_size", type=int, help="The cropped size of face", default=160)
    parser.add_argument("--minsize", type=int, help="minimum size of face", default=20)
    parser.add_argument("--mtccn_threshold", type=list, help="three model threshold", default=[0.6, 0.7, 0.7])
    parser.add_argument("--factor", type=float, help="scale factor", default=0.709)
    parser.add_argument("--margin", type=int, help="Margin for the crop around the bounding box (height, width) "
                                                   "in pixels.", default=44)
    parser.add_argument("--batch_size", type=int, help="the enqueue batch size", default=3)
    parser.add_argument("--facenet_threshold", type=float, help="Use to classify the different and the same face",
                        default=0.11)
    parser.add_argument("--file_ext", type=str, help="The ext of output file", default="png")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

