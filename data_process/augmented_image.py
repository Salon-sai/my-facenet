# -*- coding: utf-8 -*-

import os
import sys
import argparse
import shutil

from scipy import misc

import numpy as np

def main(args):
    src_dir = os.path.expanduser(args.src_dir)
    target_dir = os.path.expanduser(args.dst_dir)
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith("png"):
                src_image_path = os.path.join(root, file)
                sub_target_dir = os.path.join(target_dir, root.split("/")[-1])
                if not os.path.exists(sub_target_dir):
                    os.makedirs(sub_target_dir)

                image = misc.imread(src_image_path)
                flipped_image_path = os.path.join(sub_target_dir, "{}_flipped.png".format(file.split(".")[0]))
                flipped_image = np.fliplr(image)
                misc.imsave(flipped_image_path, flipped_image)

                if args.copy_original:
                    shutil.copy(src_image_path, sub_target_dir)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=str, help="The src directory of images")
    parser.add_argument("dst_dir", type=str, help="The target directory of augmented images and original image")
    parser.add_argument("--flip_image", type=bool, help="Do we need to flip image", default=True)
    parser.add_argument("--copy_original", type=bool, help="Do we copy the original images to target directory",
                        default=True)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))