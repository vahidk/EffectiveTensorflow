# Effective TensorFlow 2 中文版

目录
=================
## Part I: TensorFlow 2 基础
1.  [TensorFlow 2 基础](#basics)
2.  [广播](#broadcast)
3.  [利用重载OPs](#overloaded_ops)
4.  [控制流操作: 条件与循环](#control_flow)
5.  [原型核和使用Python OPs可视化](#python_ops)
6.  [TensorFlow中的数值稳定性](#stable)
---

_我们针对新发布的 TensorFlow 2.x API 更新了教程. 如果你想看 TensorFlow 1.x 的教程请移步 [v1 branch](https://github.com/vahidk/EffectiveTensorflow/tree/v1)._

_安装 TensorFlow 2.0 (alpha) 请参照 [官方网站](https://www.tensorflow.org/install/pip):_
```
pip install tensorflow==2.0.0-alpha0
```

_我们致力于逐步扩展新的文章，并保持与Tensorflow API更新同步。如果你有任何建议请提出来。_

# Part I: TensorFlow 2.0 基础
<a name="fundamentals"></a>

## TensorFlow 基础
<a name="basics"></a>
重新设计的TensorFlow 2带来了更方便使用的API。如果你熟悉numpy，你用Tensorflow 2会很爽。不像完全静态图符号计算的Tensorflow 1，TF2隐藏静态图那部分，变得像个numpy。值得注意的是，虽然交互变化了，但是TF2仍然有静态图抽象的优势，TF1能做的TF2都能做。 

让我们从一个简单的例子开始吧，我们那俩随机矩阵乘起来。我们先看看Numpy怎么做这事先。
```python
import numpy as np

x = np.random.normal(size=[10, 10])
y = np.random.normal(size=[10, 10])
z = np.dot(x, y)

print(z)
```

现在看看用TensorFlow 2.0怎么办:
```python
import tensorflow as tf

x = tf.random.normal([10, 10])
y = tf.random.normal([10, 10])
z = tf.matmul(x, y)

print(z)
```
与NumPy差不多，TensorFlow 2也马上执行并返回结果。唯一的不同是TensorFlow用tf.Tensor类型存储结果，当然这种数据可以方便的转换为NumPy数据，调用tf.Tensor.numpy()成员函数就行: 

```python
print(z.numpy())
```

为了理解符号计算的强大，让我们看看另一个例子。假设我们有从一个曲线(举个栗子 f(x) = 5x^2 + 3)上采集的样本点，并且我们要基于这些样本估计f(x)。我们建立了一个参数化函数g(x, w) = w0 x^2 + w1 x + w2，这个函数有输入x和隐藏参数w，我们的目标就是找出隐藏参数让g(x, w) ≈ f(x)。这个可以通过最小化以下的loss函数:L(w) = &sum; (f(x) - g(x, w))^2。虽然这个问题有解析解，但是我们更乐意用一个可以应用到任意可微分方程上的通用方法，嗯，SGD。我们仅需要计算L(w) 在不同样本点上关于w的平均提督，然后往梯度反方向调整就行。


那么，怎么用TensorFlow实现呢:

```python
import numpy as np
import tensorflow as tf

# 假设我们知道我们期望的多项式方程是二阶方程，
# 我们分配一个长3的向量并用随机噪声初始化。

w = tf.Variable(tf.random.normal([3, 1]))

# 用Adam优化器优化，初始学习率0.1
opt = tf.optimizers.Adam(0.1)

def model(x):
    # 定义yhat为y的估计
    f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
    yhat = tf.squeeze(tf.matmul(f, w), 1)
    return yhat

def compute_loss(y, yhat):
    # loss用y和yhat之间的L2距离估计。
    # 对w加了正则项保证w较小。
    loss = tf.nn.l2_loss(yhat - y) + 0.1 * tf.nn.l2_loss(w)
    return loss

def generate_data():
    # 根据真实函数生成一些训练样本
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
运行这段代码你会看到近似下面这个的结果:
```python
[4.9924135, 0.00040895029, 3.4504161]
```
这和我们的参数很接近了.

注意，上面的代码是交互式执行 (i.e. eager模式下ops直接执行)，这种操作并不高效. TensorFlow 2.0也提供静态图执行的法子，方便在GPUs和TPUs上快速并行执行。开启也很简单对于训练阶段函数用tf.function修饰就OK:

```python
@tf.function
def train_step():
    x, y = generate_data()

    def _loss_fn():
        yhat = model(x)
        loss = compute_loss(y, yhat)
        return loss
    
    opt.minimize(_loss_fn, [w])
```

tf.function多牛逼，他也可以吧while、for之类函数转换进去。我们后面细说。

这些只是TF能做的冰山一角。很多有几百万参数的复杂神经网络可以在TF用几行代码搞定。TF也可以在不同设备，不同线程上处理。

## 广播操作
<a name="broadcast"></a>
TF支持广播元素操作。一般来说，如果你想执行加法或者乘法之类操作，你得确保相加或者相乘元素形状匹配，比如你不能把形状为[3, 2]的tensor加到形状为[3, 4]的tensor上。但是有个特例，就是当你把一个tensor和另一有维度长度是1的tensor是去加去乘，TF会把银行的把那个维扩展，让两个tensor可操作。（去看numpy的广播机制吧）

```python
import tensorflow as tf

a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[1.], [2.]])
# c = a + tf.tile(b, [1, 2])
c = a + b

print(c)
```

广播可以让我们代码更短更高效。我们可以把不同长度的特征连接起来。比如用一些非线性操作复制特定维度，这在很多神经网络里经常用的到：


```python
a = tf.random.uniform([5, 3, 5])
b = tf.random.uniform([5, 1, 6])

# 连接a和b
tiled_b = tf.tile(b, [1, 3, 1])
c = tf.concat([a, tiled_b], 2)
d = tf.keras.layers.Dense(10, activation=tf.nn.relu).apply(c)

print(d)
```

但这个用了广播就更简单了，我们可以用f(m(x + y))等效f(mx + my)这个特性。然后隐含用广播来做连接。

```python
pa = tf.keras.layers.Dense(10).apply(a)
pb = tf.keras.layers.Dense(10).apply(b)
d = tf.nn.relu(pa + pb)

print(d)
```

事实下面的代码在可以广播的场景下更好用。

```python
def merge(a, b, units, activation=None):
    pa = tf.keras.layers.Dense(units).apply(a)
    pb = tf.keras.layers.Dense(units).apply(b)
    c = pa + pb
    if activation is not None:
        c = activation(c)
    return c
```

所以，我们说了广播的好处，那么广播有啥坏处呢。隐含的广播可能导致debug麻烦。

```python
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b)

print(c)
```

所以c的结果是啥？正确答案是12，当tensor形状不一样，TF自动的进行了广播。

避免这个问题的法子就是尽量显式，比如reduce时候注明维度。

```python
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b, 0)

print(c)
```

这里c得到[5, 7], 然后很容易发现问题。以后用reduce和tf.squeeze操作时最好注明维度。

## 利用重载函数
<a name="overloaded_ops"></a>
就像numpy，TF重载一些python操作来让graph构建更容易更可读。

切片操作可以方便的索引tensor:
```python
z = x[begin:end]  # z = tf.slice(x, [begin], [end-begin])
```
尽量不要用切片，因为这个效率很逊。为了理解这玩意效率到底有多逊，让我们康康一个例子。下面将做一个列方向上的reduce_sum。

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
我的水果Pro上执行这段花了0.045秒，好逊。这是因为执行了500次切片，很慢的，更好的法子是矩阵分解。
```python
z = tf.zeros([10])
for x_i in tf.unstack(x):
    z += x_i
```
花了0.01秒，当然，最勇的法子是用tf.reduce_sum操作:
```python
z = tf.reduce_sum(x, axis=0)
```
这个操作用了0.0001秒, 比最初的方法快了100倍。

TF也重载了一堆算数和逻辑操作
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

你也可以这些操作的扩展用法。 比如`x += y` 和 `x **= 2`。

注意，py不允许and or not之类的重载。

其他比如等于(==) 和不等(!=) 等被NumPy重载的操作并没有被TensorFlow实现，请用函数版本的 `tf.equal` 和 `tf.not_equal`。（less_equal,greater_equal之类也得用函数式）

## 控制流，条件与循环
<a name="control_flow"></a>
当我们构建一个复杂的模型，比如递归神经网络，我们需要用条件或者循环来控制操作流。这一节里我们介绍一些常用的流控制操作。

假设你想根据一个判断式来决定是否相乘或相加俩tensor。这个可以用py内置函数或者用tf.cond函数。

```python
a = tf.constant(1)
b = tf.constant(2)

p = tf.constant(True)

# 或者:
# x = tf.cond(p, lambda: a + b, lambda: a * b)
x = a + b if p else a * b

print(x.numpy())
```
由于判断式为真，因此输出相加结果，等于3。

大多数时候你在TF里用很大的tensor，并且想把操作应用到batch上。用tf.where就能对一个batch得到满足判断式的成分进行操作。
```python
a = tf.constant([1, 1])
b = tf.constant([2, 2])

p = tf.constant([True, False])

x = tf.where(p, a + b, a * b)

print(x.numpy())
```
结果得到[3, 2].

另一个常用的操作是tf.while_loop，他允许在TF里用动态循环处理可变长度序列。来个例子:

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
输出5. 注意tf.function装饰器自动把python代码转换为tf.while_loop因此我们不用折腾TF API。

现在想一下，我们想要保持完整的斐波那契数列的话，我们需要更新代码来保存历史值:
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

如果你这么执行了，TF会反馈循环值发生变化。
解决这个问题可以用 "shape invariants"，但是这个只能在底层tf.while_loop API里用。


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
这个又丑又慢。我们建立一堆没用的中间tensor。TF有更好的解决方法，用tf.TensorArray就行了:
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
TF while循环再建立负载递归神经网络时候很有用。这里有个实验，[beam search](https://en.wikipedia.org/wiki/Beam_search) 他用了tf.while_loops，你那么勇应该可以用tensor arrays实现的更高效吧。

## 原型核和用Python OPs可视化
<a name="python_ops"></a>
TF里操作kernel使用Cpp实现来保证效率。但用Cpp写TensorFlow kernel很烦诶，所以你在实现自己的kernel前可以实验下自己想法是否奏效。用tf.py_function() 你可以把任何python操作编程tf操作。

下面就是自己实现一个非线性的Relu:
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
为了验证梯度的正确性，你应该比较解析和数值梯度。
```python
# 计算解析梯度
x = tf.random.normal([10], dtype=np.float32)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = relu(x)
g = tape.gradient(y, x)
print(g)

# 计算数值梯度
dx_n = 1e-5
dy_n = relu(x + dx_n) - relu(x)
g_n = dy_n / dx_n
print(g_n)
```
这俩值应该很接近。

注意这个实现很低效，因此只应该用在原型里，因为python代码超慢，后面你会想Cpp重新实现计算kernel的，大概。

实际，我们通常用python操作来做可视化。比如你做图像分类，你在训练时想可视化你的模型预测，用Tensorboard看tf.summary.image()保存的结果吧:
```python
image = tf.placeholder(tf.float32)
tf.summary.image("image", image)
```
但是你这只能可视化输入图，没法知道预测值，用tf的操作肯定嗝屁了，你可以用python操作:
```python
def visualize_labeled_images(images, labels, max_outputs=3, name="image"):
    def _visualize_image(image, label):
        #  python里绘图
        fig = plt.figure(figsize=(3, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.imshow(image[::-1,...])
        ax.text(0, 0, str(label),
          horizontalalignment="left",
          verticalalignment="top")
        fig.canvas.draw()

        # 写入内存中
        buf = io.BytesIO()
        data = fig.savefig(buf, format="png")
        buf.seek(0)

        # Pillow解码图像
        img = PIL.Image.open(buf)
        return np.array(img.getdata()).reshape(img.size[0], img.size[1], -1)

    def _visualize_images(images, labels):
        # 只显示batch中部分图
        outputs = []
        for i in range(max_outputs):
            output = _visualize_image(images[i], labels[i])
            outputs.append(output)
        return np.array(outputs, dtype=np.uint8)

    # 、运行python op.
    figs = tf.py_function(_visualize_images, [images, labels], tf.uint8)
    return tf.summary.image(name, figs)
```

由于验证测试过一段时间测试一次，所以不用担心效率。

## Numerical stability in TensorFlow
<a name="stable"></a>
用TF或者Numpy之类数学计算库的时候，既要考虑数学计算的正确性，也要注意数值计算的稳定性。

举个例子，小学就教了x * y / y在y不等于0情况下等于x，但是实际:
```python
import numpy as np

x = np.float32(1)

y = np.float32(1e-50)  # y 被当成0了
z = x * y / y

print(z)  # prints nan
```

对于单精度浮点y太小了，直接被当成0了，当然y很大的时候也有问题:

```python
y = np.float32(1e39)  # y 被当成无穷大
z = x * y / y

print(z)  # prints nan
```

单精度浮点的最小值是1.4013e-45，任何比他小的值都被当成0，同样的任何大于3.40282e+38的,会被当成无穷大。

```python
print(np.nextafter(np.float32(0), np.float32(1)))  # prints 1.4013e-45
print(np.finfo(np.float32).max)  # print 3.40282e+38
```
为了保证你计算的稳定，你必须避免过小值或者过大值。这个听起来理所当然，但是在TF进行梯度下降的时候可能很难debug。你在FP时候要保证稳定，在BP时候还要保证。

让我们看一个例子，我们想要在一个logits向量上计算softmax，一个naive的实现就像：
```python
import tensorflow as tf

def unstable_softmax(logits):
    exp = tf.exp(logits)
    return exp / tf.reduce_sum(exp)

print(unstable_softmax([1000., 0.]).numpy())  # prints [ nan, 0.]
```
所以你logits的exp的值，即使logits很小会得到很大的值，说不定超过单精度的范围。最大的不溢出logit值是ln(3.40282e+38) = 88.7，比他大的就会导致nan。

所以怎么让这玩意稳定，exp(x - c) / &sum; exp(x - c) = exp(x) / &sum; exp(x)就搞掂了。如果我们logits减去一个数，结果还是一样的，一般减去logits最大值。这样exp函数的输入被限定在[-inf, 0]，然后输出就是[0.0, 1.0]，就很棒:

```python
import tensorflow as tf

def softmax(logits):
    exp = tf.exp(logits - tf.reduce_max(logits))
    return exp / tf.reduce_sum(exp)

print(softmax([1000., 0.]).numpy())  # prints [ 1., 0.]
```

我们看一个更加复杂的情况，考虑一个分类问题，我们用softmax来得到logits的可能性，之后用交叉熵计算预测和真值。交叉熵这么算xe(p, q) = -&sum; p_i log(q_i)。然后一个naive的实现如下:

```python
def unstable_softmax_cross_entropy(labels, logits):
    logits = tf.math.log(softmax(logits))
    return -tf.reduce_sum(labels * logits)

labels = tf.constant([0.5, 0.5])
logits = tf.constant([1000., 0.])

xe = unstable_softmax_cross_entropy(labels, logits)

print(xe.numpy())  # prints inf
```

由于softmax输出结果接近0，log的输出接近无限导致了计算的不稳定，我们扩展softmax并简化了计算交叉熵:

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

我们也证明了梯度计算的正确性:
```python
with tf.GradientTape() as tape:
    tape.watch(logits)
    xe = softmax_cross_entropy(labels, logits)
    
g = tape.gradient(xe, logits)
print(g.numpy())  # prints [0.5, -0.5]
```
这就对了。

必须再次提醒，在做梯度相关操作时候必须注意保证每一层梯度都在有效范围内，exp和log操作由于可以把小数变得很大，因此可能让计算变得不稳定，所以使用exp和log操作必须十分谨慎。
