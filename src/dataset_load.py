from config import *
import os

import tensorflow as tf


def _parse_function(proto):
    f = {
        "x": tf.FixedLenSequenceFeature([(NUM_MFCCS-1)*MAX_SEQUENCE_LENGTH], tf.float32, default_value=0., allow_missing=True),
        "label": tf.FixedLenSequenceFeature([MAX_SEQUENCE_LENGTH], tf.int64, default_value=0, allow_missing=True)
    }
    parsed_features = tf.parse_single_example(proto, f)
    return parsed_features['x'], parsed_features['label']


def create_tfrecords_iterator(input_path, batch_size, shuffle_buffer):
    """
    Create an iterator over the TFRecords file with chroma features.

    :param input_path: can accept both a file and a folder
    :param batch_size:
    :param shuffle_buffer: if None, don't shuffle
    :return: dataset.make_one_shot_iterator()
    """
    if os.path.isdir(input_path):
        data_file = [os.path.join(input_path, fp) for fp in os.listdir(input_path)]
    elif os.path.isfile(input_path):
        data_file = input_path
    else:
        raise ValueError("please specify a valid path, folder or file")
    dataset = tf.data.TFRecordDataset(data_file)
    if shuffle_buffer is None:
        dataset = dataset.map(_parse_function, num_parallel_calls=16).repeat().batch(batch_size)
    else:
        dataset = dataset.map(_parse_function, num_parallel_calls=16).shuffle(shuffle_buffer).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator()
