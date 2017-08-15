# Tensorflow training framework

Follow the example in dataset/mnist.py and model/convnet_classifier.py for
examples of how to define custom datasets and models.

## Install dependencies
```
pip install tensorflow numpy pillow matplotlib six
```

## Preparing datasets
Currently the framework includes code for preprocessing mnist, cifar10, and cifar100 datasets.

To download and preprocess the mnist dataset run:
```
python -m dataset.mnist convert
```

Run the following to visualize an example:
```
python -m dataset.mnist visualize
```

In the above snippets you could replace mnist with cifar10 or cifar100 to preprocess the respective datasets.

## Training
To train an mnist classification model run:
```
python -m main --model=convnet_classifier --dataset=mnist
```

To visualize the training logs on Tensorboard run:
```
tensorboard --logdir=output
```
