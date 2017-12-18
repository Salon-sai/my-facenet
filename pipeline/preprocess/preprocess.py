# -*- coding: utf-8 -*-

import argparse
import sys
import os

from scipy import misc
import tensorflow as tf
import numpy as np

import pipeline.preprocess.align.detect_face as detect_face

def main(args):
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    margin = args.margin

    root_dir = os.path.expanduser(args.data_dir)
    src_dir = os.path.join(root_dir, "src")
    preprocess_dir = os.path.join(root_dir, "preprocess")

    with tf.Graph().as_default():
        with tf.Session() as session:
            pnet, rnet, onet = detect_face.create_mtcnn(session, None)
            for family_id in os.listdir(src_dir):
                family_dir = os.path.join(src_dir, family_id)
                target_dir = os.path.join(preprocess_dir, family_id)
                for image_name in os.listdir(family_dir):
                    nrof_images_total += 1
                    # 生成照片名称相对应的pre-process目录
                    image_base_name = os.path.splitext(image_name)[0]
                    pp_image_dir = os.path.join(target_dir, image_base_name)

                    image_path = os.path.join(family_dir, image_name)
                    image = misc.imread(image_path, mode='RGB')
                    bounding_boxes, _ = detect_face.detect_face(image, args.minsize, pnet, rnet, onet,
                                                                args.threshold, args.factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces > 0:
                        # det = bounding_boxes[:, 0: 4]
                        # det_arr = []
                        if not os.path.isdir(pp_image_dir):
                            os.makedirs(pp_image_dir)
                        img_size = np.asarray(image.shape)[0:2]
                        for index, bounding_box in enumerate(bounding_boxes):
                            bounding_box = np.squeeze(bounding_box)
                            bb = np.zeros(4, dtype=np.int32)
                            # The square records the origin face square
                            square = (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])
                            bb[0] = np.maximum(bounding_box[0] - margin / 2, 0) # x1
                            bb[1] = np.maximum(bounding_box[1] - margin / 2, 0) # y1
                            bb[2] = np.minimum(bounding_box[2] + margin / 2, img_size[1])   # x2
                            bb[3] = np.minimum(bounding_box[3] + margin / 2, img_size[0])   # y2

                            cropped = image[bb[1]:bb[3],bb[0]:bb[2], :]
                            aligned = misc.imresize(cropped, (args.face_size, args.face_size), interp="bilinear")
                            nrof_successfully_aligned += 1
                            output_filename = "{}_{}_{}_{}".format(image_base_name, square, index, "." + args.file_ext)
                            misc.imsave(os.path.join(pp_image_dir, output_filename), aligned)
                    else:
                        print("can not extract the face from the image : %s" % image_path)
            print("Total number of images: %d" % nrof_images_total)
            print("Number of successfully aligned images: %d" % nrof_images_total)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, help="the directory of data set")
    # parser.add_argument('model_dir', type=str, help="the directory of model used as face detection")
    parser.add_argument("--minsize", type=int, help="minimum size of face", default=20)
    parser.add_argument("--threshold", type=list, help="three model threshold", default=[0.6, 0.7, 0.7])
    parser.add_argument("--factor", type=float, help="scale factor", default=0.709)
    parser.add_argument("--face_size", type=int, help="the size of each cropped faces", default=224)
    parser.add_argument("--margin", type=int, help="Margin for the crop around the bounding box (height, width) "
                                                   "in pixels." , default=44)
    parser.add_argument("--file_ext", type=str, help="The extension of output image files", default="png")

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
