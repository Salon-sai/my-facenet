# -*- coding: utf-8 -*-

import argparse

import tensorflow as tf

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help="Indicates if a new classifier should be trained or a classification", default='CLASSIFY')

    
    return parser.parse_args(argv)
