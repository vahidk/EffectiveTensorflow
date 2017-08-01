# Tensorflow training framework

Folow the example in dataset/mnist.py and model/convnet_classifier.py for
examples of how to define custom models and datasets.

## Mnist
To download mnist dataset run:
```
python -m dataset.mnist convert
```

Run the following to visualize an example:
```
python -m dataset.mnist visualize
```

## Usage
To train an mnist classification model run:
```
python -m main --model=convnet_classifier --dataset=mnist
```

To visualize the training logs in Tensorboard run:
```
tensorboard --logdir=outptut
```
