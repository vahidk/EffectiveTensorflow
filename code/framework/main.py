from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from common import ops
import dataset.mnist
import model.convnet_classifier

tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_string('model', 'convnet_classifier', 'Model name.')
tf.flags.DEFINE_string('dataset', 'mnist', 'Dataset name.')
tf.flags.DEFINE_string('output_dir', '', 'Optional output dir.')
tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule.')
tf.flags.DEFINE_string('hparams', '', 'Hyper parameters.')
tf.flags.DEFINE_integer('save_summary_steps', 10, 'Summary steps.')
tf.flags.DEFINE_integer('save_checkpoints_steps', 10, 'Checkpoint steps.')
tf.flags.DEFINE_integer('eval_steps', None, 'Number of eval steps.')
tf.flags.DEFINE_integer('eval_frequency', 10, 'Eval frequency.')
tf.flags.DEFINE_integer('num_gpus', 0, 'Numner of gpus.')

FLAGS = tf.flags.FLAGS
learn = tf.contrib.learn

MODELS = {
  'convnet_classifier': model.convnet_classifier
}

DATASETS = {
  'mnist': dataset.mnist
}

HPARAMS = {
  'optimizer': 'Adam',
  'learning_rate': 0.001,
  'decay_steps': 10000,
  'batch_size': 128
}

def get_hparams():
  hparams = HPARAMS
  hparams.update(DATASETS[FLAGS.dataset].HPARAMS)
  hparams.update(MODELS[FLAGS.model].HPARAMS)

  hparams = tf.contrib.training.HParams(**hparams)
  hparams.parse(FLAGS.hparams)

  return hparams


def make_input_fn(mode, params):
  def _input_fn():
    with tf.device(tf.DeviceSpec(device_type='CPU', device_index=0)):
      dataset = DATASETS[FLAGS.dataset]
      tensors = learn.read_batch_features(
        file_pattern=dataset.get_split(mode),
        batch_size=params.batch_size,
        features=dataset.FEATURES,
        reader=tf.TFRecordReader,
        randomize_input=True if mode == learn.ModeKeys.TRAIN else False,
        num_epochs=None if mode == learn.ModeKeys.TRAIN else 1,
        queue_capacity=params.batch_size*3,
        feature_queue_capacity=params.batch_size*2,
        reader_num_threads=8 if mode == learn.ModeKeys.TRAIN else 1)
      features, labels = dataset.map_features(tensors)
    return features, labels
  return _input_fn


def make_model_fn():
  def _model_fn(features, labels, mode, params):
    model_fn = MODELS[FLAGS.model].model_fn

    global_step = tf.train.get_or_create_global_step()

    if FLAGS.num_gpus > 0 and mode == learn.ModeKeys.TRAIN:
      split_features = {k: tf.split(v, FLAGS.num_gpus)
                        for k, v in features.iteritems()}
      split_labels = {k: tf.split(v, FLAGS.num_gpus)
                      for k, v in labels.iteritems()}
      grads = []
      predictions = collections.defaultdict(list)
      losses = []

      opt = ops.create_optimizer(
        params.optimizer, params.learning_rate, params.decay_steps)

      for i in range(FLAGS.num_gpus):
        with tf.device(tf.DeviceSpec(device_type='GPU', device_index=i)):
          with tf.name_scope('tower_%d' % i):
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
              device_features = {k: v[i] for k, v in split_features.iteritems()}
              device_labels = {k: v[i] for k, v in split_labels.iteritems()}

              device_predictions, device_loss = model_fn(
                device_features, device_labels, mode, params)

              for k, v in device_predictions.iteritems():
                predictions[k].append(v)

              if device_loss is not None:
                losses.append(device_loss)

              device_grads = opt.compute_gradients(device_loss)
              grads.append(device_grads)

      grads = ops.average_gradients(grads)
      train_op = opt.apply_gradients(grads, global_step=global_step)

      for k, v in predictions.iteritems():
        predictions[k] = tf.concat(v, axis=0)

      loss = tf.add_n(losses) if losses else None
    else:
      with tf.device(tf.DeviceSpec(device_type='GPU', device_index=0)):
        predictions, loss = model_fn(features, labels, mode, params)

        train_op = None
        if mode == learn.ModeKeys.TRAIN:
          opt = ops.create_optimizer(
            params.optimizer, params.learning_rate, params.decay_steps)
          train_op = opt.minimize(loss, global_step=global_step)

    tf.summary.scalar('loss/loss', loss)

    return tf.contrib.learn.ModelFnOps(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op)

  return _model_fn


def experiment_fn(run_config, hparams):
  estimator = learn.Estimator(
    model_fn=make_model_fn(), config=run_config, params=hparams)
  eval_metrics = MODELS[FLAGS.model].eval_metrics_fn(hparams)
  return learn.Experiment(
    estimator=estimator,
    train_input_fn=make_input_fn(learn.ModeKeys.TRAIN, hparams),
    eval_input_fn=make_input_fn(learn.ModeKeys.EVAL, hparams),
    eval_metrics=eval_metrics,
    eval_steps=FLAGS.eval_steps,
    min_eval_frequency=FLAGS.eval_frequency)


def main(unused_argv):
  if FLAGS.output_dir:
    model_dir = FLAGS.output_dir
  else:
    model_dir = 'output/%s_%s' % (FLAGS.model, FLAGS.dataset)
  session_config = tf.ConfigProto()
  session_config.allow_soft_placement = True
  session_config.gpu_options.allow_growth = True
  run_config = learn.RunConfig(
    model_dir=model_dir,
    save_summary_steps=FLAGS.save_summary_steps,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    save_checkpoints_secs=None,
    session_config=session_config)

  estimator = learn.learn_runner.run(
    experiment_fn=experiment_fn,
    run_config=run_config,
    schedule=FLAGS.schedule,
    hparams=get_hparams())


if __name__ == '__main__':
  tf.app.run()
