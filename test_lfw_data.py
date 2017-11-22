# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import data_flow_ops

import data_process as dp
import itertools

def image_decode(filename, image_size=200):
    file_contents = tf.read_file(filename)
    image = tf.image.decode_image(file_contents, channels=3)
    image = tf.random_crop(image, [image_size, image_size, 3])
    image.set_shape((image_size, image_size, 3))
    return tf.image.per_image_standardization(image)

def main():
    BATCH_SIZE = 90

    with tf.Graph().as_default():

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name="image_paths")

        batch_size_placeholder = tf.placeholder(tf.int32, name="batch_size")

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string],
                                              shapes=[(3,)],
                                              shared_name=None, name=None)

        enqueue_op = input_queue.enqueue_many([image_paths_placeholder])

        nrof_preprocess_threads = 4
        all_images = []
        for _ in range(nrof_preprocess_threads):
            filenames = input_queue.dequeue()
            images = [image_decode(filename) for filename in tf.unstack(filenames)]
            all_images.append(images)

        image_batch = tf.train.batch_join(
            tensors_list=all_images, batch_size=batch_size_placeholder,
            capacity=4411,
            # shapes=[(200, 200, 3)],
            enqueue_many=True,
            allow_smaller_final_batch=True
        )

        image_batch = tf.identity(image_batch, 'image_batch')

        dataset = dp.get_dataset('~/data/lfw')

        dataset = list(itertools.chain(*dataset))

        image_paths = np.reshape(dataset, (-1, 3))

        init_local_op = tf.local_variables_initializer()

        with tf.Session() as session:
            session.run(init_local_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            session.run(enqueue_op, feed_dict={image_paths_placeholder: image_paths})

            batch_num = int(np.ceil(len(image_paths) / BATCH_SIZE))
            for i in range(batch_num):
                batch_size = min(BATCH_SIZE, len(image_paths) - i * BATCH_SIZE)
                images = session.run(image_batch, feed_dict={batch_size_placeholder: batch_size})

            coord.request_stop()
            coord.join(threads=threads)

if __name__ == '__main__':
    main()

