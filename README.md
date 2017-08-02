# Effective Tensorflow

Table of Contents
=================
1. [Tensorflow Basics](#basics)
2. [Understanding static and dynamic shapes](#shapes)
3. [Broadcasting the good and the ugly](#broadcast)
4. [Prototyping kernels and advanced visualization with Python ops](#python_ops)

## Tensorflow Basics
<a name="basics"></a>
The most striking difference between Tensorflow and other numerical computation libraries such as numpy is that operations in Tensorflow are symbolic. This is a powerful concept that allows Tensorflow to do all sort of things (e.g. automatic differentiation) that are not possible with imperative libraries such as numpy. But it also comes at the cost of making it harder to grasp. Our attempt here is demystify Tensorflow and provide some guidelines and best practices for more effective use of Tensorflow.

Let's start with a simple example, we want to multiply two random matrices. First we look at an implementation done in numpy:
```python
import numpy as np

x = np.random.normal(size=[10, 10])
y = np.random.normal(size=[10, 10])
z = np.dot(x, y)

print(z)
```

Now we perform the exact same computation this time in Tensorflow:
```python
import tensorflow as tf

x = tf.random_normal([10, 10])
y = tf.random_normal([10, 10])
z = tf.matmul(x, y)

sess = tf.Session()
z_val = sess.run(z)

print(z_val)
```
Unlike numpy that immediately performs the computation and copies the result to
the output variable z, tensorflow only gives us a handle (of type Tensor) to a node in the graph that represents the result. If we try printing the value of z directly, we get something like this:
```
Tensor("MatMul:0", shape=(10, 10), dtype=float32)
```
Since both the inputs have a fully defined shape, tensorflow is able to infer the shape of the tensor as well as its type. In order to compute the value of the tensor we need to create a session and evaluate it using Session.run method.

***
__Tip__: When using Jupyter notebook make sure to call tf.reset_default_graph() at the beginning to clear the symbolic graph before defining new nodes.
***

To understand how powerful symbolic computation can be let's have a look at another example. Assume that we have samples from a curve (say f(x) = 5x^2 + 3) and we want to estimate f(x) without knowing its parameters. We define a parametric function g(x, w) = w0 x^2 + w1 x + w2, which is a function of the input x and latent parameters w, our goal is then to find the latent parameters such that g(x, w) ≈ f(x). This can be done by minimizing the following loss function: L(w) = (f(x) - g(x, w))^2. Although there's a closed form solution for this simple problem, we opt to use a more general approach that can be applied to any arbitrary differentiable function, and that is using stochastic gradient descent. We simply compute the average gradient of L(w) with respect to w over a set of sample points and move in the opposite direction. 

Here's how it can be done in Tensorflow:

```python
import numpy as np
import tensorflow as tf

# Placeholders are used to feed values from python to Tensorflow ops. We define
# two placeholders, one for input feature x, and one for output y.
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Assuming we know that the desired function is a polynomial of 2nd degree, we
# allocate a vector of size 3 to hold the coefficients. The variable will be
# automatically initialized with random noise.
w = tf.get_variable("w", shape=[3, 1])

# We define yhat to be our estimate of y.
f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
yhat = tf.squeeze(tf.matmul(f, w), 1)

# The loss is defined to be the l2 distance between our estimate of y and its
# true value. We also added a shrinkage term, tp ensure the resulting weights 
# would be small.
loss = tf.nn.l2_loss(yhat - y) + 0.1 * tf.nn.l2_loss(w)

# We use the Adam optimizer with learning rate set to 0.1 to minimize the loss.
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

def generate_data():
    x_val = np.random.uniform(-10.0, 10.0, size=100)
    y_val = 5 * np.square(x_val) + 3
    return x_val, y_val

sess = tf.Session()
# Since we are using variables we first need to initialize them.
sess.run(tf.global_variables_initializer())
for _ in range(1000):
    x_val, y_val = generate_data()
    _, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})
    print(loss_val)
print(sess.run([w]))
```
By running this piece of code you should see a result close to this:
```
[4.9924135, 0.00040895029, 3.4504161]
```
Which is a relatively close approximation to our parameters. 

This is just tip of the iceberg for what Tensorflow can do. Many problems such a optimizing large neural networks with millions of parameters can be implemented efficiently in Tensorflow in just a few lines of code. Tensorflow takes care of scaling across multiple devices, and threads, and supports a variety of platforms.

For simplicity in most of the examples here we manually create sessions and we don't care about saving and loading checkpoints but this is not how we usually do things in practice. You most probably want to use the estimator API to take care of session management and logging. We provide a simple extendable framework in the code/framework directory for an example of a practical framework for training neural networks using Tensorflow.

## Understanding static and dynamic shapes
<a name="shapes"></a>
Tensors in Tensorflow have a static shape attribute which is determined during graph construction. The static shape may be underspecified. For example we might define a tensor of shape [None, 128]:
```python
import tensorflow as tf

a = tf.placeholder([None, 128])
```
This means that the first dimension can be of any size and will be determined dynamically during Session.run. Tensorflow has a rather ugly API for exposing the static shape:
```python
static_shape = a.get_shape().as_list()  # returns [None, 128]
```
(This used to be a.shape but someone decided it's too convenient.)

To get the dynamic shape of the tensor you can call tf.shape op, which returns a tensor representing the shape of the given tensor:
```python
dynamic_shape = tf.shape(a)
```

The static shape of a tensor can be set with Tensor.set_shape() method:
```python
a.set_shape([32, 128])
```
Use this function only if you know what you are doing, in practice it's safer to do dynamic reshaping with tf.reshape() op:
```python
a =  tf.reshape(a, [32, 128])
```

It can be convenient to have a function that returns the static shape when available and dynamic shape when it's not. The following utility function does just that:
```python
def get_shape(tensor):
  static_shape = tensor.get_shape().as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims
```

Now imagine we want to convert a Tensor of rank 3 to a tensor of rank 2 by collapsing the second and third dimensions into one. We can use our get_shape() function to do that:
```python
b = placeholder([None, 10, 32])
shape = get_shape(tensor)
b = tf.reshape(b, [shape[0], shape[1] * shape[2]])
```
Note that this works whether the shapes are statically specified or not.

In fact we can write a general purpose reshape function to collapse any list of dimensions:
```python
import tensorflow as tf
import numpy as np

def reshape(tensor, dims_list):
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
```

Then collapsing the second dimension becomes very easy:
```python
b = placeholder([None, 10, 32])
b = tf.reshape(b, [0, [1, 2]])
```

## Broadcasting the good and the ugly
<a name="broadcast"></a>
Tensorflow supports broadcasting elementwise operations. Normally when you want to perform operations like addition and multiplication, you need to make sure that shapes of the operands match, e.g. you can’t add a tensor of shape [3, 2] to a tensor of shape [3, 4]. But there’s a special case and that’s when you have a singular dimension. Tensorflow implicitly tiles the tensor across its singular dimensions to match the shape of the other operand. So it’s valid to add a tensor of shape [3, 2] to a tensor of shape [3, 1]

```python
import tensorflow as tf

a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[1.], [2.]])
# c = a + tf.tile(a, [1, 2])
c = a + b 
```

Broadcasting allows us to perform implicit tiling which makes the code shorter, and more memory efficient, since we don’t need to store the result of the tiling operation. One neat place that this can be used is when combining features of different length. In order to concatenate features of different length we commonly tile the input tensors, concatenate the result and apply some nonlinearity. This is a common pattern across a variety of neural network architectures:

```python
a = tf.random_uniform([5, 3, 5])
b = tf.random_uniform([5, 1, 6])

# concat a and b and apply nonlinearity
tiled_b = tf.tile(b, [1, 3, 1])
c = tf.concat([a, tiled_b], 2)
d = tf.layers.dense(c, 10, activation=tf.nn.relu)
```

But this can be done more efficiently with broadcasting. We use the fact that f(m(x + y)) is equal to f(mx + my). So we can do the linear operations separately and use broadcasting to do implicit concatenation:

```python
pa = tf.layers.dense(a, 10, activation=None)
pb = tf.layers.dense(b, 10, activation=None)
d = tf.nn.relu(pa + pb)
```

In fact this piece of code is pretty general and can be applied to tensors of arbitrary shape as long as broadcasting between tensors is possible:

```python
def tile_concat_dense(a, b, units, activation=tf.nn.relu):
    pa = tf.layers.dense(a, units, activation=None)
    pb = tf.layers.dense(b, units, activation=None)
    c = pa + pb
    if activation is not None:
        c = activation(c)
    return c
```

So far we discussed the good part of broadcasting. But what’s the ugly part you may ask? Implicit assumptions almost always make debugging harder to do. Consider the following example:

```python
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b)
```

What do you think would the value of c would after evaluation? If you guessed 6, that’s wrong. It’s going to be 12. This is because when rank of two tensors don’t match, Tensorflow automatically expands the first dimension of the tensor with lower rank before the elementwise operation, so the result of addition would be [[2, 3], [3, 4]], and the reducing over all parameters would give us 12.

The way to avoid this problem is to be as explicit as possible. Had we specified which dimension we would want to reduce across, catching this bug would have been much easier:

```python
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b, 0)
```

Here the value of c would be [5, 7], and we immediately would guess based on the shape of the result that there’s something wrong. A general rule of thumb is to always specify the dimensions in reduction operations and when using tf.squeeze.

## Prototyping kernels and advanced visualization with Python ops
<a name="python_ops"></a>
Operation kernels in Tensorflow are entirely written in C++ for efficiency. But writing a Tensorflow kernel in C++ can be quite a pain. So, before spending hours implementing your kernel you may want to prototype something quickly, however inefficient. With tf.py_func() you can turn any piece of python code to a Tensorflow operation.

For example this is how you can implement a simple ReLU nonlinearity kernel in Tensorflow as a python op:
```python
import numpy as np
import tensorflow as tf
import uuid

def relu(inputs):
    # Define the op in python
    def _relu(x):
        return np.maximum(x, 0.)
    
    # Define the op's gradient in python
    def _relu_grad(x):
        return np.float32(x > 0)

    # An adapter that defines a gradient op compatible with Tensorflow
    def _relu_grad_op(op, grad):
        x = op.inputs[0]
        x_grad = grad * tf.py_func(_relu_grad, [x], tf.float32)
        return x_grad
    
    # Register the gradient with a unique id
    grad_name = "MyReluGrad_" + str(uuid.uuid4())
    tf.RegisterGradient(grad_name)(_relu_grad_op)

    # Override the gradient of the custom op
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        output = tf.py_func(_relu, [inputs], tf.float32)
    return output
```

To verify that the gradients are correct you can use Tensorflow's gradient checker:
```python
x = tf.random_normal([10])
y = relu(x * x)

with tf.Session():
    diff = tf.test.compute_gradient_error(x, [10], y, [10])
    print(diff)
```
compute_gradient_error() computes the gradient numerically and returns the difference between the provided gradient. What we want is a very low difference.

Note that this implementation is pretty inefficient, and is only useful for prototyping, since the python code is not parallelizable and won't run on GPU. Once you verified your idea, you definitely would want to write it as a C++ kernel.

In practice we commonly use python ops to do visualization on Tensorboard. Consider the case that you are building an image classification model and want to visualize your model predictions during training. Tensorflow allows visualizing images with tf.summary.image() function:
```python
image = tf.placeholder(tf.float32)
tf.summary.image("image", image)
```
But this only visualizes the input image. In order to visualize the predictions you have to find a way to annotated the image which may be almost impossible to do in Tensorflow. An easier way to do this is to do the drawing in python, and wrap it in a python op:
```python
import io
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

def visualize_labeled_images(images, labels, max_outputs=3, name='image'):
    def _visualize_image(image, label):
        # Do the actual drawing in python
        fig = plt.figure(figsize=(3, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.imshow(image[::-1,...])
        ax.text(0, 0, str(label), 
          horizontalalignment='left', 
          verticalalignment='top')
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
```

Note that since summaries are usually only evaluated once in a while (not per step), this implementation may be used in practice without worrying about efficiency.
