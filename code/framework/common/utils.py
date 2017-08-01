from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mp
import tensorflow as tf


def parallel_record_writer(iterator, create_example, path, num_threads=8):
  """Create a RecordIO file from data for efficient reading."""

  def _queue(inputs):
    for item in iterator:
      inputs.put(item)
    for _ in range(num_threads):
      inputs.put(None)

  def _map_fn(inputs, outputs):
    while True:
      item = inputs.get()
      if item is None:
        break
      example = create_example(item)
      outputs.put(example)
    outputs.put(None)

  # Read the inputs.
  inputs = mp.Queue()
  mp.Process(target=_queue, args=(inputs,)).start()

  # Convert to tf.Example
  outputs = mp.Queue()
  for _ in range(num_threads):
    mp.Process(target=_map_fn, args=(inputs, outputs)).start()

  # Write the output to file.
  writer = tf.python_io.TFRecordWriter(path)
  counter = 0
  while True:
    example = outputs.get()
    if example is None:
      counter += 1
      if counter == num_threads:
        break
      else:
        continue
    writer.write(example.SerializeToString())
  writer.close()
