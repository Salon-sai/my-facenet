# -*- coding: utf-8 -*-

import argparse
import sys
import os
import datetime

import age_model, gender_model
from data_process.imdb_process import ImageDatabase, calculate_embedding
from data_process.xiaoyu_process import ImageAgeDatabase

def main(args):
    imdb_dir = os.path.expanduser(args.imdb_aligned_root)
    if args.train_attr == "GENDER":
        image_database = ImageDatabase(imdb_dir, args.max_num_images)
    elif args.train_attr == "AGE":
        image_database = ImageAgeDatabase(imdb_dir)
    else:
        raise ValueError('Invalid training properties')

    batch_size = args.batch_size
    now_datetime = datetime.datetime.now()

    gender_subdir = datetime.datetime.strftime(now_datetime, 'gender-%Y%m%d-%H%M%S')
    gender_log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), gender_subdir)
    gender_model_dir = os.path.join(os.path.expanduser(args.models_base_dir), gender_subdir)

    age_subdir = datetime.datetime.strftime(now_datetime, 'age-%Y%m%d-%H%M%S')
    age_log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), age_subdir)
    age_model_dir = os.path.join(os.path.expanduser(args.models_base_dir), age_subdir)

    image_database.embeddings = calculate_embedding(facenet_model_path=args.model_path,
                                                    images_path=image_database.images_path,
                                                    batch_size=batch_size,
                                                    image_size=args.image_size)
    embedding_size = image_database.embeddings.shape[1]

    if args.train_attr == "GENDER":
        if not os.path.isdir(gender_log_dir):
            os.makedirs(gender_log_dir)

        if not os.path.isdir(gender_model_dir):
            os.makedirs(gender_model_dir)

        gender_model.gender_classifier(embedding_size, args.weight_decay_l1, args.learning_rate, args.learning_rate_decay_step,
                          args.learning_rate_decay_factor, args.optimizer, args.epoch_size, args.batch_size,
                          gender_log_dir, gender_model_dir, gender_subdir, image_database)
    elif args.train_attr == "AGE":
        if not os.path.isdir(age_log_dir):
            os.makedirs(age_log_dir)

        if not os.path.isdir(age_model_dir):
            os.makedirs(age_model_dir)

        age_model.age_classifier(embedding_size,  args.weight_decay_l1, args.learning_rate, args.learning_rate_decay_step,
                       args.learning_rate_decay_factor, args.optimizer, args.epoch_size, args.batch_size,
                       age_log_dir, age_model_dir, age_subdir, image_database)
    else:
        raise ValueError('Invalid training properties')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help="The model of calculating the embedding vector")
    parser.add_argument('--imdb_aligned_root', type=str, help='The directory of imdb cropped and aligned data',
                        default="~/data/imdb_cropped_clean")
    parser.add_argument("--save_embedding_path", type=str, help="The directory saving embeddings",
                        default="~/data/imdb_cropped_clean_embedd")
    parser.add_argument("--batch_size", type=int, help="The size of train batch", default=100)
    parser.add_argument("--epoch_size", type=int, help="The size of epoch size", default=500)
    parser.add_argument("--image_size", type=int, help="The size of image", default=160)
    parser.add_argument("--max_num_images", type=int, help="The max number of images used to training, valid and test",
                        default=10000)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument("--learning_rate", type=float, help="training learning rate", default=0.1)
    parser.add_argument("--learning_rate_decay_step", type=int,
                        help="Number of global step between learning rate decay.", default=10000)
    parser.add_argument('--learning_rate_decay_factor', type=float, help='Learning rate decay factor.', default=0.9)
    parser.add_argument("--logs_base_dir", type=str, help='Directory where to write event logs.', default='logs/')
    parser.add_argument("--models_base_dir", type=str, help="Direcotry where to save the parameters of model",
                        default="models/")
    parser.add_argument("--weight_decay_l1", type=float, help="L1 weight regularization", default=0.0)
    parser.add_argument("--train_attr", type=str, choices=["GENDER", "AGE"],
                        help="which property will be trained", default="GENDER")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
