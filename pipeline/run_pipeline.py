# -*- coding: utf-8 -*-

import sys
import argparse

from pipeline.preprocess import preprocess
from pipeline.process import process

def main(args):
    preprocess_dir = preprocess.call_preprocess(data_dir=args.data_dir, src_sub_dir=args.src_subdir,
                               preprocess_sub_dir=args.preprocess_subdir)
    process.call_process(data_dir=args.data_dir, preprocess_sub_dir=args.preprocess_subdir,
                         process_subdir=args.process_subdir, save_subdir=args.save_subdir,
                         identification_model_dir=args.identification_model_dir,
                         gender_model_dir=args.gender_model_dir, age_model_dir=args.age_model_dir)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, help="the directory of data set")
    # parser.add_argument('model_dir', type=str, help="the directory of model used as face detection")
    parser.add_argument("--src_subdir", type=str, help="The sub directory of data set src", default="src")
    parser.add_argument("--preprocess_subdir", type=str, help="The sub directory of data set preprocess",
                        default="preprocess")
    parser.add_argument("--process_subdir", type=str, help="The sub directory of data set process", default="process")
    parser.add_argument("--save_subdir", type=str, help="The sub directory of data set save", default="save")

    parser.add_argument("--identification_model_dir", type=str, help="The directory of facenet cnn model",
                        default="./models/20170512-110547/20170512-110547.pb")
    parser.add_argument("--gender_model_dir", type=str, help="The directory of gender model",
                        default='./models/gender-20180115-144702')
    parser.add_argument("--age_model_dir", type=str, help="The directory of age model",
                        default='./models/age-20180124-142925')

    parser.add_argument("--minsize", type=int, help="minimum size of face", default=20)
    parser.add_argument("--threshold", nargs="+", type=float, help="three model threshold", default=[0.6, 0.7, 0.7])
    parser.add_argument("--factor", type=float, help="scale factor", default=0.709)
    parser.add_argument("--face_size", type=int, help="the size of each cropped faces", default=224)
    parser.add_argument("--margin", type=int, help="Margin for the crop around the bounding box (height, width) "
                                                   "in pixels." , default=44)
    parser.add_argument("--file_ext", type=str, help="The extension of output image files", default="png")

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))