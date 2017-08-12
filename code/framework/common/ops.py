from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_shape(tensor):
  """Returns static shape if available and dynamic shape otherwise."""
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims


def reshape(tensor, dims_list):
  """Reshape the given tensor by collapsing dimensions."""
  shape = get_shape(tensor)
  dims_prod = []
  for dims in dims_list:
    if isinstance(dims, int):
      dims_prod.append(shape[dims])
    elif all([isinstance(shape[d], int) for d in dims]):
      dims_prod.append(np.prod([shape[d] for d in dims]))
    else:
      dims_prod.append(tf.prod([shape[d] for d in dims]))
  tensor = tf.reshape(tensor, dims_prod)
  return tensor


def dense_layers(tensor,
                 sizes,
                 activation=tf.nn.relu,
                 linear_top_layer=False,
                 drop_rate=0.0,
                 name=None,
                 **kwargs):
  """Builds a stack of fully connected layers with optional dropout."""
  with tf.variable_scope(name, default_name='dense_layers'):
    for i, size in enumerate(sizes):
      if i == len(sizes) - 1 and linear_top_layer:
        activation = None
      tensor = tf.layers.dropout(tensor, drop_rate)
      tensor = tf.layers.dense(
          tensor,
          size,
          name='dense_layer_%d' % i,
          activation=activation,
          **kwargs)
  return tensor


def conv_layers(tensor,
                filters,
                kernels,
                pools,
                padding="same",
                activation=tf.nn.relu,
                drop_rate=0.0,
                **kwargs):
  for fs, ks, ps in zip(filters, kernels, pools):
    tensor = tf.layers.dropout(tensor, drop_rate)
    tensor = tf.layers.conv2d(
        tensor,
        filters=fs,
        kernel_size=ks,
        padding=padding,
        activation=activation,
        **kwargs)
    if ps and ps > 1:
      tensor = tf.layers.max_pooling2d(
        inputs=tensor, pool_size=ps, strides=ps, padding=padding)
  return tensor


def create_optimizer(optimizer, learning_rate, decay_steps=None, **kwargs):
  global_step = tf.train.get_or_create_global_step()

  if decay_steps:
    learning_rate = tf.train.exponential_decay(
      learning_rate, global_step, decay_steps, 0.5, staircase=True)

  return tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer](
    learning_rate, **kwargs)


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = [g for g, _ in grad_and_vars]
    grad = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)
    v = grad_and_vars[0][1]
    average_grads.append((grad, v))
  return average_grads
