"""Utility functions for visualization on tensorboard."""

import io
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf


def labeled_image(name, images, labels, max_outputs=3, flip_vertical=False,
                  color='pink', font_size=15):
    def _visualize_image(image, label):
        # Do the actual drawing in python
        fig = plt.figure(figsize=(3, 3), dpi=80)
        ax = fig.add_subplot(111)
        if flip_vertical:
            image = image[::-1,...]
        ax.imshow(image.squeeze())
        ax.text(0, 0, str(label),
          horizontalalignment='left',
          verticalalignment='top',
          color=color,
          fontsize=font_size)
        fig.canvas.draw()

        # Write the plot as a memory file.
        buf = io.BytesIO()
        data = fig.savefig(buf, format='png')
        buf.seek(0)

        # Read the image and convert to numpy array
        img = PIL.Image.open(buf)
        return np.array(img.getdata()).reshape(img.size[0], img.size[1], -1)

    def _visualize_images(images, labels):
        # Only display the given number of examples in the batch
        outputs = []
        for i in range(max_outputs):
            output = _visualize_image(images[i], labels[i])
            outputs.append(output)
        return np.array(outputs, dtype=np.uint8)

    # Run the python op.
    figs = tf.py_func(_visualize_images, [images, labels], tf.uint8)
    return tf.summary.image(name, figs)
