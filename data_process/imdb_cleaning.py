# -*- coding: utf-8 -*-

import argparse
import sys
import os
import threading
import queue

import numpy as np
import scipy.io as sio

from scipy import misc
from datetime import datetime
from datetime import timedelta

person_images = queue.Queue(maxsize=1000000)
mutex = threading.Lock()

def main(args):
    threads = []
    imdb_path = os.path.expanduser(args.imdb_path)
    output_path = os.path.expanduser(args.output_dir)

    load_data_mat(imdb_path, args.min_score, args.only_single_face)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    def worker():
        global person_images
        while not person_images.empty():
            person_image = person_images.get()
            image = misc.imread(person_image.image_full_path)
            ext_splits = person_image.image_path.split(".")
            save_path = os.path.join(output_path, person_image.image_path.split("/")[0])
            if mutex.acquire():
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                mutex.release()
            filename = "{}_{}_{}.{}".format(ext_splits[0], person_image.age, person_image.gender, ext_splits[1])
            misc.imsave(os.path.join(output_path, filename), image)
            person_images.task_done()

    for i in range(args.num_multi_thread):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()


def crop_image(person_image):
    image = misc.imread(person_image.image_full_path)
    try:
        return image[person_image._face_location[1]: person_image._face_location[3],
                                  person_image._face_location[0]: person_image._face_location[2], :]
    except IndexError:
        return image


def load_data_mat(imdb_path, mini_score, only_single_face):
    global person_images
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

    # the face score larger than min score and gender must be defined
    face_ids = np.where(np.logical_and(face_scores > mini_score, np.logical_not(np.isnan(gender_info))))[0]
    single_face_ids = np.argwhere(np.isnan(second_scores)).flatten()    # only use the has single face

    if only_single_face:
        indexes = list(set(face_ids) & set(single_face_ids))
    else:
        indexes = face_ids

    for i, ii in enumerate(indexes):
        if len(image_path_info[ii]) > 1:
            print(image_path_info)
        try:
            person_image = PersonImage(
                data_dir=imdb_path,
                image_path=image_path_info[ii][0],
                dob=dob_info[ii],
                taken_date=photo_taken_info[ii],
                gender=gender_info[ii],
                name=cele_names[ii],
                face_location=face_locations[ii],
                only_single_face=only_single_face
            )
            person_images.put(person_image)
        except OverflowError:
            print("The birth date of dob: %d, the image path: %s" %
                  (dob_info[ii], os.path.join(imdb_path, image_path_info[ii][0])))

class PersonImage(object):

    def __init__(self, data_dir, image_path, dob, taken_date, gender, name, face_location, only_single_face):
        self._name = name
        self._gender = int(gender)
        self._original_data_path = data_dir
        self._image_path = image_path
        self._taken_date = taken_date
        self._birth_date = datetime.fromordinal(int(dob)) + timedelta(days=int(dob) % 1) - timedelta(days=366)
        self._age = self._taken_date - self._birth_date.year
        self._face_location = np.asarray(face_location[0], dtype=np.int32)

    @property
    def image_full_path(self):
        return os.path.join(self._original_data_path, self._image_path)

    @property
    def image_path(self):
        return self._image_path

    @property
    def age(self):
        return self._age

    @property
    def gender(self):
        return self._gender

    @property
    def face_location(self):
        return self._face_location

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("imdb_path", type=str, help="The imdb data path")
    parser.add_argument("output_dir", type=str, help="output directory of saving clean image")
    parser.add_argument("--min_score", type=float, help="The first face mini score to filter low quality", default=2.5)
    parser.add_argument("--only_single_face", type=bool, help="Choose only one face in images", default=True)
    parser.add_argument("--num_multi_thread", type=int, help="The number of multi-threads which process crop and "
                                                             "save image", default=4)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

