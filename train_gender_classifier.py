# -*- coding: utf-8 -*-

import argparse
import sys

import tensorflow as tf

def main(args):
    pass

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help="The model of calculating the embedding vector")
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help="Indicates if a new classifier should be trained or a classification", default='CLASSIFY')
    parser.add_argument('--train_data_dir', type=str, help='train data directory', default="~/data/imdb_corp")

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
