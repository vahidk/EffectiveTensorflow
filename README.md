# Effective TensorFlow 2

Table of Contents
=================
## Part I: TensorFlow 2 Fundamentals
1.  [TensorFlow 2 Basics](#basics)
2.  [Broadcasting the good and the ugly](#broadcast)
3.  [Take advantage of the overloaded operators](#overloaded_ops)
4.  [Control flow operations: conditionals and loops](#control_flow)
5.  [Prototyping kernels and advanced visualization with Python ops](#python_ops)
6.  [Numerical stability in TensorFlow](#stable)
---

_We updated the guide to follow the newly released TensorFlow 2.x API. If you want the original guide for TensorFlow 1.x see the [v1 branch](https://github.com/vahidk/EffectiveTensorflow/tree/v1)._

_To install TensorFlow 2.0 (alpha) follow the [instructions on the official website](https://www.tensorflow.org/install/pip):_
```
pip install tensorflow==2.0.0-alpha0
```

_We aim to gradually expand this series by adding new articles and keep the content up to date with the latest releases of TensorFlow API. If you have suggestions on how to improve this series or find the explanations ambiguous, feel free to create an issue, send patches, or reach out by email._
```

# Part I: TensorFlow 2.0 Fundamentals
<a name="fundamentals"></a>

## TensorFlow Basics
<a name="basics"></a>
TensorFlow 2 went under a massive redesign to make the API more accessible and easier to use. If you are familiar with numpy you will find yourself right at home when using TensorFlow 2. Unlike TensorFlow 1 which was purely symbolic, TensorFlow 2 hides its symbolic nature behind the hood to look like any other imperative library like NumPy. It's important to note the change is mostly an interface change, and TensorFlow 2 is still able to take advantage of its symbolic machinery to do everything that TensorFlow 1.x can do (e.g. automatic-differentiation and massively parallel computation on TPUs/GPUs).

Let's start with a simple example, we want to multiply two random matrices. First we look at an implementation done in NumPy:
```python
import numpy as np

x = np.random.normal(size=[10, 10])
y = np.random.normal(size=[10, 10])
z = np.dot(x, y)

print(z)
```

Now we perform the exact same computation this time in TensorFlow 2.0:
```python
import tensorflow as tf

x = tf.random.normal([10, 10])
y = tf.random.normal([10, 10])
z = tf.matmul(x, y)

print(z)
```
Similar to NumPy TensorFlow 2 also immediately performs the computation and produces the result. The only difference is that TensorFlow uses tf.Tensor type to store the results which can be easily converted to NumPy, by calling tf.Tensor.numpy() member function: 

```
print(z.numpy())
```

To understand how powerful symbolic computation can be let's have a look at another example. Assume that we have samples from a curve (say f(x) = 5x^2 + 3) and we want to estimate f(x) based on these samples. We define a parametric function g(x, w) = w0 x^2 + w1 x + w2, which is a function of the input x and latent parameters w, our goal is then to find the latent parameters such that g(x, w) ≈ f(x). This can be done by minimizing the following loss function: L(w) = &sum; (f(x) - g(x, w))^2. Although there's a closed form solution for this simple problem, we opt to use a more general approach that can be applied to any arbitrary differentiable function, and that is using stochastic gradient descent. We simply compute the average gradient of L(w) with respect to w over a set of sample points and move in the opposite direction.

Here's how it can be done in TensorFlow:

```python
import numpy as np
import tensorflow as tf

# Assuming we know that the desired function is a polynomial of 2nd degree, we
# allocate a vector of size 3 to hold the coefficients and initialize it with
# random noise.
w = tf.Variable(tf.random.normal([3, 1]))

# We use the Adam optimizer with learning rate set to 0.1 to minimize the loss.
opt = tf.optimizers.Adam(0.1)

def model(x):
    # We define yhat to be our estimate of y.
    f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
    yhat = tf.squeeze(tf.matmul(f, w), 1)
    return yhat

def compute_loss(y, yhat):
    # The loss is defined to be the l2 distance between our estimate of y and its
    # true value. We also added a shrinkage term, to ensure the resulting weights
    # would be small.
    loss = tf.nn.l2_loss(yhat - y) + 0.1 * tf.nn.l2_loss(w)
    return loss

def generate_data():
    # Generate some training data based on the true function
    x = np.random.uniform(-10.0, 10.0, size=100).astype(np.float32)
    y = 5 * np.square(x) + 3
    return x, y

def train_step():
    x, y = generate_data()

    def _loss_fn():
        yhat = model(x)
        loss = compute_loss(y, yhat)
        return loss
    
    opt.minimize(_loss_fn, [w])

for _ in range(1000):
    train_step()

print(w.numpy())
```
By running this piece of code you should see a result close to this:
```
[4.9924135, 0.00040895029, 3.4504161]
```
Which is a relatively close approximation to our parameters.

Note that in the above code we are running Tensorflow in imperative mode (i.e. operations get instantly executed), which is not very efficient. TensorFlow 2.0 can also turn a given piece of python code into a graph which can then optimized and efficiently parallelized on GPUs and TPUs. To get all those benefits we simply need to decorate the train_step function with tf.function decorator:

```
@tf.function
def train_step():
    x, y = generate_data()

    def _loss_fn():
        yhat = model(x)
        loss = compute_loss(y, yhat)
        return loss
    
    opt.minimize(_loss_fn, [w])
```

What's cool about tf.function is that it's also able to convert basic python statements like while, for and if into native TensorFlow functions. We will get to that later.

This is just tip of the iceberg for what TensorFlow can do. Many problems such as optimizing large neural networks with millions of parameters can be implemented efficiently in TensorFlow in just a few lines of code. TensorFlow takes care of scaling across multiple devices, and threads, and supports a variety of platforms.

## Broadcasting the good and the ugly
<a name="broadcast"></a>
TensorFlow supports broadcasting elementwise operations. Normally when you want to perform operations like addition and multiplication, you need to make sure that shapes of the operands match, e.g. you can’t add a tensor of shape [3, 2] to a tensor of shape [3, 4]. But there’s a special case and that’s when you have a singular dimension. TensorFlow implicitly tiles the tensor across its singular dimensions to match the shape of the other operand. So it’s valid to add a tensor of shape [3, 2] to a tensor of shape [3, 1]

```python
import tensorflow as tf

a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[1.], [2.]])
# c = a + tf.tile(b, [1, 2])
c = a + b

print(c)
```

Broadcasting allows us to perform implicit tiling which makes the code shorter, and more memory efficient, since we don’t need to store the result of the tiling operation. One neat place that this can be used is when combining features of varying length. In order to concatenate features of varying length we commonly tile the input tensors, concatenate the result and apply some nonlinearity. This is a common pattern across a variety of neural network architectures:

```python
a = tf.random.uniform([5, 3, 5])
b = tf.random.uniform([5, 1, 6])

# concat a and b and apply nonlinearity
tiled_b = tf.tile(b, [1, 3, 1])
c = tf.concat([a, tiled_b], 2)
d = tf.keras.layers.Dense(10, activation=tf.nn.relu).apply(c)

print(d)
```

But this can be done more efficiently with broadcasting. We use the fact that f(m(x + y)) is equal to f(mx + my). So we can do the linear operations separately and use broadcasting to do implicit concatenation:

```python
pa = tf.keras.layers.Dense(10).apply(a)
pb = tf.keras.layers.Dense(10).apply(b)
d = tf.nn.relu(pa + pb)

print(d)
```

In fact this piece of code is pretty general and can be applied to tensors of arbitrary shape as long as broadcasting between tensors is possible:

```python
def merge(a, b, units, activation=None):
    pa = tf.keras.layers.Dense(units).apply(a)
    pb = tf.keras.layers.Dense(units).apply(b)
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

print(c)
```

What do you think the value of c would be after evaluation? If you guessed 6, that’s wrong. It’s going to be 12. This is because when rank of two tensors don’t match, TensorFlow automatically expands the first dimension of the tensor with lower rank before the elementwise operation, so the result of addition would be [[2, 3], [3, 4]], and the reducing over all parameters would give us 12.

The way to avoid this problem is to be as explicit as possible. Had we specified which dimension we would want to reduce across, catching this bug would have been much easier:

```python
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b, 0)

print(c)
```

Here the value of c would be [5, 7], and we immediately would guess based on the shape of the result that there’s something wrong. A general rule of thumb is to always specify the dimensions in reduction operations and when using tf.squeeze.

## Take advantage of the overloaded operators
<a name="overloaded_ops"></a>
Just like NumPy, TensorFlow overloads a number of python operators to make building graphs easier and the code more readable.

The slicing op is one of the overloaded operators that can make indexing tensors very easy:
```python
z = x[begin:end]  # z = tf.slice(x, [begin], [end-begin])
```
Be very careful when using this op though. The slicing op is very inefficient and often better avoided, especially when the number of slices is high. To understand how inefficient this op can be let's look at an example. We want to manually perform reduction across the rows of a matrix:
```python
import tensorflow as tf
import time

x = tf.random.uniform([500, 10])

z = tf.zeros([10])

start = time.time()
for i in range(500):
    z += x[i]
print("Took %f seconds." % (time.time() - start))
```
On my MacBook Pro, this took 0.045 seconds to run which is quite slow. The reason is that we are calling the slice op 500 times, which is going to be very slow to run. A better choice would have been to use tf.unstack op to slice the matrix into a list of vectors all at once:
```python
z = tf.zeros([10])
for x_i in tf.unstack(x):
    z += x_i
```
This took 0.01 seconds. Of course, the right way to do this simple reduction is to use tf.reduce_sum op:
```python
z = tf.reduce_sum(x, axis=0)
```
This took 0.0001 seconds, which is 300x faster than the original implementation.

TensorFlow also overloads a range of arithmetic and logical operators:
```python
z = -x  # z = tf.negative(x)
z = x + y  # z = tf.add(x, y)
z = x - y  # z = tf.subtract(x, y)
z = x * y  # z = tf.mul(x, y)
z = x / y  # z = tf.div(x, y)
z = x // y  # z = tf.floordiv(x, y)
z = x % y  # z = tf.mod(x, y)
z = x ** y  # z = tf.pow(x, y)
z = x @ y  # z = tf.matmul(x, y)
z = x > y  # z = tf.greater(x, y)
z = x >= y  # z = tf.greater_equal(x, y)
z = x < y  # z = tf.less(x, y)
z = x <= y  # z = tf.less_equal(x, y)
z = abs(x)  # z = tf.abs(x)
z = x & y  # z = tf.logical_and(x, y)
z = x | y  # z = tf.logical_or(x, y)
z = x ^ y  # z = tf.logical_xor(x, y)
z = ~x  # z = tf.logical_not(x)
```

You can also use the augmented version of these ops. For example `x += y` and `x **= 2` are also valid.

Note that Python doesn't allow overloading "and", "or", and "not" keywords.

Other operators that aren't supported are equal (==) and not equal (!=) operators which are overloaded in NumPy but not in TensorFlow. Use the function versions instead which are `tf.equal` and `tf.not_equal`.

## Control flow operations: conditionals and loops
<a name="control_flow"></a>
When building complex models such as recurrent neural networks you may need to control the flow of operations through conditionals and loops. In this section we introduce a number of commonly used control flow ops.

Let's assume you want to decide whether to multiply to or add two given tensors based on a predicate. This can be simply implemented with either python's built-in if statement or using tf.cond function:
```python
a = tf.constant(1)
b = tf.constant(2)

p = tf.constant(True)

# Alternatively:
# x = tf.cond(p, lambda: a + b, lambda: a * b)
x = a + b if p else a * b

print(x.numpy())
```
Since the predicate is True in this case, the output would be the result of the addition, which is 3.

Most of the times when using TensorFlow you are using large tensors and want to perform operations in batch. A related conditional operation is tf.where, which like tf.cond takes a predicate, but selects the output based on the condition in batch.
```python
a = tf.constant([1, 1])
b = tf.constant([2, 2])

p = tf.constant([True, False])

x = tf.where(p, a + b, a * b)

print(x.numpy())
```
This will return [3, 2].

Another widely used control flow operation is tf.while_loop. It allows building dynamic loops in TensorFlow that operate on sequences of variable length. Let's see how we can generate Fibonacci sequence with tf.while_loops:

```python
@tf.function
def fibonacci(n):
    a = tf.constant(1)
    b = tf.constant(1)

    for i in range(2, n):
        a, b = b, a + b
    
    return b
    
n = tf.constant(5)
b = fibonacci(n)
    
print(b.numpy())
```
This will print 5. Note that tf.function automatically converts the given python code to use tf.while_loop so we don't need to directly interact with the TF API.

Now imagine we want to keep the whole series of Fibonacci sequence. We may update our body to keep a record of the history of current values:
```python
@tf.function
def fibonacci(n):
    a = tf.constant(1)
    b = tf.constant(1)
    c = tf.constant([1, 1])

    for i in range(2, n):
        a, b = b, a + b
        c = tf.concat([c, [b]], 0)
    
    return c
    
n = tf.constant(5)
b = fibonacci(n)
    
print(b.numpy())
```

Now if you try running this, TensorFlow will complain that the shape of the the one of the loop variables is changing. 
One way to fix this is is to use "shape invariants", but this functionality is only available when using the low-level tf.while_loop API:


```python
n = tf.constant(5)

def cond(i, a, b, c):
    return i < n

def body(i, a, b, c):
    a, b = b, a + b
    c = tf.concat([c, [b]], 0)
    return i + 1, a, b, c

i, a, b, c = tf.while_loop(
    cond, body, (2, 1, 1, tf.constant([1, 1])),
    shape_invariants=(tf.TensorShape([]),
                      tf.TensorShape([]),
                      tf.TensorShape([]),
                      tf.TensorShape([None])))

print(c.numpy())
```

This is not only getting ugly, but is also pretty inefficient. Note that we are building a lot of intermediary tensors that we don't use. TensorFlow has a better solution for this kind of growing arrays. Meet tf.TensorArray. Let's do the same thing this time with tensor arrays:
```python
@tf.function
def fibonacci(n):
    a = tf.constant(1)
    b = tf.constant(1)

    c = tf.TensorArray(tf.int32, n)
    c = c.write(0, a)
    c = c.write(1, b)

    for i in range(2, n):
        a, b = b, a + b
        c = c.write(i, b)
    
    return c.stack()

n = tf.constant(5)
c = fibonacci(n)
    
print(c.numpy())
```
TensorFlow while loops and tensor arrays are essential tools for building complex recurrent neural networks. As an exercise try implementing [beam search](https://en.wikipedia.org/wiki/Beam_search) using tf.while_loops. Can you make it more efficient with tensor arrays?

## Prototyping kernels and advanced visualization with Python ops
<a name="python_ops"></a>
Operation kernels in TensorFlow are entirely written in C++ for efficiency. But writing a TensorFlow kernel in C++ can be quite a pain. So, before spending hours implementing your kernel you may want to prototype something quickly, however inefficient. With tf.py_function() you can turn any piece of python code to a TensorFlow operation.

For example this is how you can implement a simple ReLU nonlinearity kernel in TensorFlow as a python op:
```python
import numpy as np
import tensorflow as tf
import uuid

def relu(inputs):
    # Define the op in python
    def _py_relu(x):
        return np.maximum(x, 0.)

    # Define the op's gradient in python
    def _py_relu_grad(x):
        return np.float32(x > 0)
    
    @tf.custom_gradient
    def _relu(x):
        y = tf.py_function(_py_relu, [x], tf.float32)
        
        def _relu_grad(dy):
            return dy * tf.py_function(_py_relu_grad, [x], tf.float32)

        return y, _relu_grad

    return _relu(inputs)
```

To verify that the gradients are correct you can compare the numerical and analytical gradients and compare the vlaues.
```python
# Compute analytical gradient
x = tf.random.normal([10], dtype=np.float32)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = relu(x)
g = tape.gradient(y, x)
print(g)

# Compute numerical gradient
dx_n = 1e-5
dy_n = relu(x + dx_n) - relu(x)
g_n = dy_n / dx_n
print(g_n)
```
The numbers should be very close.

Note that this implementation is pretty inefficient, and is only useful for prototyping, since the python code is not parallelizable and won't run on GPU. Once you verified your idea, you definitely would want to write it as a C++ kernel.

In practice we commonly use python ops to do visualization on Tensorboard. Consider the case that you are building an image classification model and want to visualize your model predictions during training. TensorFlow allows visualizing images with tf.summary.image() function:
```python
image = tf.placeholder(tf.float32)
tf.summary.image("image", image)
```
But this only visualizes the input image. In order to visualize the predictions you have to find a way to add annotations to the image which may be almost impossible with existing ops. An easier way to do this is to do the drawing in python, and wrap it in a python op:
```python
def visualize_labeled_images(images, labels, max_outputs=3, name="image"):
    def _visualize_image(image, label):
        # Do the actual drawing in python
        fig = plt.figure(figsize=(3, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.imshow(image[::-1,...])
        ax.text(0, 0, str(label),
          horizontalalignment="left",
          verticalalignment="top")
        fig.canvas.draw()

        # Write the plot as a memory file.
        buf = io.BytesIO()
        data = fig.savefig(buf, format="png")
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
    figs = tf.py_function(_visualize_images, [images, labels], tf.uint8)
    return tf.summary.image(name, figs)
```

Note that since summaries are usually only evaluated once in a while (not per step), this implementation may be used in practice without worrying about efficiency.

## Numerical stability in TensorFlow
<a name="stable"></a>
When using any numerical computation library such as NumPy or TensorFlow, it's important to note that writing mathematically correct code doesn't necessarily lead to correct results. You also need to make sure that the computations are stable.

Let's start with a simple example. From primary school we know that x * y / y is equal to x for any non zero value of x. But let's see if that's always true in practice:
```python
import numpy as np

x = np.float32(1)

y = np.float32(1e-50)  # y would be stored as zero
z = x * y / y

print(z)  # prints nan
```

The reason for the incorrect result is that y is simply too small for float32 type. A similar problem occurs when y is too large:

```python
y = np.float32(1e39)  # y would be stored as inf
z = x * y / y

print(z)  # prints nan
```

The smallest positive value that float32 type can represent is 1.4013e-45 and anything below that would be stored as zero. Also, any number beyond 3.40282e+38, would be stored as inf.

```python
print(np.nextafter(np.float32(0), np.float32(1)))  # prints 1.4013e-45
print(np.finfo(np.float32).max)  # print 3.40282e+38
```

To make sure that your computations are stable, you want to avoid values with small or very large absolute value. This may sound very obvious, but these kind of problems can become extremely hard to debug especially when doing gradient descent in TensorFlow. This is because you not only need to make sure that all the values in the forward pass are within the valid range of your data types, but also you need to make sure of the same for the backward pass (during gradient computation).

Let's look at a real example. We want to compute the softmax over a vector of logits. A naive implementation would look something like this:
```python
import tensorflow as tf

def unstable_softmax(logits):
    exp = tf.exp(logits)
    return exp / tf.reduce_sum(exp)

print(unstable_softmax([1000., 0.]).numpy())  # prints [ nan, 0.]
```
Note that computing the exponential of logits for relatively small numbers results to gigantic results that are out of float32 range. The largest valid logit for our naive softmax implementation is ln(3.40282e+38) = 88.7, anything beyond that leads to a nan outcome.

But how can we make this more stable? The solution is rather simple. It's easy to see that exp(x - c) / &sum; exp(x - c) = exp(x) / &sum; exp(x). Therefore we can subtract any constant from the logits and the result would remain the same. We choose this constant to be the maximum of logits. This way the domain of the exponential function would be limited to [-inf, 0], and consequently its range would be [0.0, 1.0] which is desirable:

```python
import tensorflow as tf

def softmax(logits):
    exp = tf.exp(logits - tf.reduce_max(logits))
    return exp / tf.reduce_sum(exp)

print(softmax([1000., 0.]).numpy())  # prints [ 1., 0.]
```

Let's look at a more complicated case. Consider we have a classification problem. We use the softmax function to produce probabilities from our logits. We then define our loss function to be the cross entropy between our predictions and the labels. Recall that cross entropy for a categorical distribution can be simply defined as xe(p, q) = -&sum; p_i log(q_i). So a naive implementation of the cross entropy would look like this:

```python
def unstable_softmax_cross_entropy(labels, logits):
    logits = tf.math.log(softmax(logits))
    return -tf.reduce_sum(labels * logits)

labels = tf.constant([0.5, 0.5])
logits = tf.constant([1000., 0.])

xe = unstable_softmax_cross_entropy(labels, logits)

print(xe.numpy())  # prints inf
```

Note that in this implementation as the softmax output approaches zero, the log's output approaches infinity which causes instability in our computation. We can rewrite this by expanding the softmax and doing some simplifications:

```python
def softmax_cross_entropy(labels, logits):
    scaled_logits = logits - tf.reduce_max(logits)
    normalized_logits = scaled_logits - tf.reduce_logsumexp(scaled_logits)
    return -tf.reduce_sum(labels * normalized_logits)

labels = tf.constant([0.5, 0.5])
logits = tf.constant([1000., 0.])

xe = softmax_cross_entropy(labels, logits)

print(xe.numpy())  # prints 500.0
```

We can also verify that the gradients are also computed correctly:
```python
with tf.GradientTape() as tape:
    tape.watch(logits)
    xe = softmax_cross_entropy(labels, logits)
    
g = tape.gradient(xe, logits)
print(g.numpy())  # prints [0.5, -0.5]
```
which is correct.

Let me remind again that extra care must be taken when doing gradient descent to make sure that the range of your functions as well as the gradients for each layer are within a valid range. Exponential and logarithmic functions when used naively are especially problematic because they can map small numbers to enormous ones and the other way around.

