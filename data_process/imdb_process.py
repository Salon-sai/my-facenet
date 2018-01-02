# -*- coding: utf-8 -*-

import argparse
import sys
import os

import numpy as np
import scipy.io as sio

from scipy import misc
from datetime import datetime
from datetime import timedelta

def main(args):
    imdb_path = os.path.expanduser(args.imdb_path)
    mini_score = args.min_score
    person_images = load_data_mat(imdb_path, mini_score, args.only_single_face)
    print(len(person_images))


def load_data_mat(imdb_path, mini_score, only_single_face):
    mat_file = os.path.join(imdb_path, "imdb.mat")
    mat_data = sio.loadmat(mat_file)
    imdb_info = mat_data['imdb'][0, 0]
    dob_info = imdb_info[0][0]              # The information of date of birth (Matlab serial date number)
    photo_taken_info = imdb_info[1][0]      # year when the photo was taken
    image_path_info = imdb_info[2][0]       # full path with image
    gender_info = imdb_info[3][0]           # gender info of each image (0 for female and 1 for male, NaN if unknown)
    cele_names = imdb_info[4][0]            # the image of celebrity names
    face_locations = imdb_info[5][0]        # face location of image
    face_scores = imdb_info[6][0]           # face score
    second_scores = imdb_info[7][0]         # the score of second face in image

    high_quality_face_ids = np.argwhere(face_scores > mini_score).flatten()
    single_face_ids = np.argwhere(np.isnan(second_scores)).flatten()    # only use the has single face

    if only_single_face:
        indexes = list(set(high_quality_face_ids) & set(single_face_ids))
    else:
        indexes = high_quality_face_ids

    person_images = []
    for i, ii in enumerate(indexes):
        if len(image_path_info[ii]) > 1:
            print(image_path_info)
        person_images.append(
            PersonImage(
                image_path=os.path.join(imdb_path, image_path_info[ii][0]),
                dob=dob_info[ii],
                taken_date=photo_taken_info[ii],
                gender=gender_info[ii],
                name=cele_names[ii],
                face_location=face_locations[ii],
                only_single_face=only_single_face
            )
        )

    return person_images

class PersonImage(object):

    def __init__(self, image_path, dob, taken_date, gender, name, face_location, only_single_face):
        self._name = name
        self._gender = gender,
        self._image_path = image_path
        self._taken_date = taken_date
        try:
            self._birth_date = datetime.fromordinal(int(dob)) + timedelta(days=int(dob) % 1) - timedelta(days=366)
            self._age = self._taken_date - self._birth_date.year
        except OverflowError:
            print("The birth date of dob: %d, the image path: %s" % (dob, image_path))
        self._face_location = np.asarray(face_location[0], dtype=np.int32)
        if len(face_location) > 1:
            print(name)
        self._load_image()


    def _load_image(self):
        image = misc.imread(self._image_path)
        try:
            self.crop_image = image[self._face_location[1]: self._face_location[3],
                               self._face_location[0]: self._face_location[2], :]
        except IndexError:
            print("The image cannot exact the faces, the image path is %s and "
                  "the location: [%d, %d, %d, %d]" % (self._image_path, self._face_location[0], self._face_location[1],
                                                      self._face_location[2], self._face_location[3]))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("imdb_path", type=str, help="The imdb data path")
    parser.add_argument("--min_score", type=float, help="The first face mini score to filter low quality", default=2.5)
    parser.add_argument("--only_single_face", type=bool, help="Choose only one face in images", default=True)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

