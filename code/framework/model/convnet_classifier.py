from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from common import ops
from common import summary

FLAGS = tf.flags.FLAGS

HPARAMS = {
  'drop_rate': 0.5,
  'crop_margin': 8,
}


def model_fn(features, labels, mode, params):
  images = features['image']
  labels = labels['label']

  drop_rate = params.drop_rate if mode == tf.estimator.ModeKeys.TRAIN else 0.0

  features = ops.conv_layers(
    images,
    filters=[32, 64, 128],
    kernels=[3, 3, 3],
    pools=[2, 2, 2])

  features = tf.contrib.layers.flatten(features)

  logits = ops.dense_layers(
    features, [512, params.num_classes],
    drop_rate=drop_rate,
    linear_top_layer=True)

  predictions = tf.argmax(logits, axis=1)

  loss = tf.losses.sparse_softmax_cross_entropy(
    labels=labels, logits=logits)

  summary.labeled_image("images", images, predictions)

  return {'predictions': predictions}, loss


def eval_metrics_fn(params):
  metrics_dict = {}
  metrics_dict['accuracy'] = tf.contrib.learn.MetricSpec(tf.metrics.accuracy)
  return metrics_dict
