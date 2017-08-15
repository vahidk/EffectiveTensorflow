"""Cifar100 dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
from six.moves import cPickle
from six.moves import urllib
import struct
import sys
import tarfile
import tensorflow as tf

from common import utils

REMOTE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
LOCAL_DIR = os.path.join('data/cifar100/')
ARCHIVE_NAME = 'cifar-100-python.tar.gz'
DATA_DIR = 'cifar-100-python/'
TRAIN_BATCHES = ['train']
TEST_BATCHES = ['test']

IMAGE_SIZE = 32
NUM_CLASSES = 100


FEATURES = {
  'image': tf.FixedLenFeature([], tf.string),
  'label': tf.FixedLenFeature([], tf.int64),
}

HPARAMS = {
  'image_size': IMAGE_SIZE,
  'num_classes': NUM_CLASSES,
}


def get_split(split):
  """Returns train/test split paths."""
  output_data = os.path.join(LOCAL_DIR, 'data_%s.tfrecord' % split)
  return output_data


def map_features(features):
  """Adapts read data to model input."""
  def _decode_image(image):
    image = tf.to_float(tf.image.decode_image(image, channels=3)) / 255.0
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    return image

  image = tf.map_fn(_decode_image, features['image'], tf.float32)
  label = features['label']
  return {'image': image}, {'label': label}


def _download_data():
  """Download the cifar dataset."""
  if not os.path.exists(LOCAL_DIR):
    os.makedirs(LOCAL_DIR)
  if not os.path.exists(LOCAL_DIR + ARCHIVE_NAME):
    print('Downloading...')
    urllib.request.urlretrieve(REMOTE_URL, LOCAL_DIR + ARCHIVE_NAME)
  if not os.path.exists(LOCAL_DIR + DATA_DIR):
    print('Extracting files...')
    tar = tarfile.open(LOCAL_DIR + ARCHIVE_NAME)
    tar.extractall(LOCAL_DIR)
    tar.close()


def _image_iterator(split):
  """An iterator that reads and returns images and labels from cifar."""
  batches = {
    tf.estimator.ModeKeys.TRAIN: TRAIN_BATCHES,
    tf.estimator.ModeKeys.EVAL: TEST_BATCHES
  }[split]

  for batch in batches:
    with open('%s%s%s' % (LOCAL_DIR, DATA_DIR, batch), 'rb') as fo:
      dict = cPickle.load(fo)
      images = np.array(dict['data'])
      labels = np.array(dict['fine_labels'])

      num = images.shape[0]
      images = np.reshape(images, [num, 3, IMAGE_SIZE, IMAGE_SIZE])
      images = np.transpose(images, [0, 2, 3, 1])
      print('Loaded %d examples.' % num)

      for i in range(num):
        yield utils.encode_image(images[i]), labels[i]


def _convert_data(split):
  """Convert the dataset to TFRecord format."""
  def _create_example(item):
    image, label = item
    example = tf.train.Example(features=tf.train.Features(
      feature={
        'image': tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[label.astype(np.int64)]))
      }))
    return example

  utils.parallel_record_writer(
    _image_iterator(split), _create_example, get_split(split))


def _visulize_data(split=tf.estimator.ModeKeys.TRAIN):
  """Read an visualize the first example form the dataset."""
  path = get_split(split)
  iterator = tf.python_io.tf_record_iterator(path)
  item = next(iterator)

  example = tf.train.Example()
  example.ParseFromString(item)

  image = utils.decode_image(
    example.features.feature['image'].bytes_list.value[0])
  label = example.features.feature['label'].int64_list.value[0]

  plt.imshow(image.astype(np.uint8))
  plt.title('Label: %d' % label)
  plt.show()


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Usage: python dataset.cifar100 <convert|visualize>')
    sys.exit(1)

  if sys.argv[1] == 'convert':
    _download_data()
    _convert_data(tf.estimator.ModeKeys.TRAIN)
    _convert_data(tf.estimator.ModeKeys.EVAL)
  elif sys.argv[1] == 'visualize':
    _visulize_data()
  else:
    print('Unknown command', sys.argv[1])
