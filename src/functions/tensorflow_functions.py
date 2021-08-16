"""Implementação das diferentes funções de benchmark em Tensorflow."""

import numpy as np
import tensorflow as tf

import src.functions.core as core


class F1(core.Function):
  """Sphere (First function of De Jong) [Molga and Smutnicki 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.12, max=5.12)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    return tf.reduce_sum(x * x, axis=0)


class F2(core.Function):
  """Rosenbrock (Second Function of De Jong) [Molga and Smutnicki 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    xi = x[:-1]
    xnext = x[1:]
    return tf.reduce_sum(100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2, axis=0)


class F3(core.Function):
  """SumSquares (Axis Parallel Hyper-Ellipsoid) [Molga and Smutnicki 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    mul = tf.range(1, x.shape[0] + 1, dtype=x.dtype)
    return tf.reduce_sum((x ** 2) * mul, axis=0)


class F4(core.Function):
  """Rotated Hyper-Ellipsoid [Molga and Smutnicki 2005]."""

  def __init__(self,
               domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    x = tf.cast(x, tf.float32)
    d = x.shape[0]

    return tf.reduce_sum(tf.convert_to_tensor(
      [tf.reduce_sum(x[0:(i + 1)] ** 2, axis=0) for i in range(d)],
      dtype=tf.float32), axis=0)


class F5(core.Function):
  """Ackley [Molga and Smutnicki 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-32.768, max=32.768),
               a=20,
               b=0.2, c=2 * np.math.pi):
    super().__init__(domain)
    self._a = tf.convert_to_tensor(a, dtype=tf.float32)
    self._b = tf.convert_to_tensor(b, dtype=tf.float32)
    self._c = tf.convert_to_tensor(c, dtype=tf.float32)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    d = x.shape[0]
    return -self.a * tf.exp(
      -self.b * tf.sqrt(tf.reduce_sum(x * x, axis=0) / d)) - \
           tf.exp(
             tf.reduce_sum(tf.cos(self.c * x), axis=0) / d) + self.a + np.math.e

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  @property
  def c(self):
    return self._c


class F6(core.Function):
  """Levy [Laguna and Martí 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    pi = np.math.pi
    d = x.shape[0] - 1
    w = 1 + (x - 1) / 4

    term1 = tf.sin(pi * w[0]) ** 2
    term3 = (w[d] - 1) ** 2 * (1 + tf.sin(2 * pi * w[d]) ** 2)

    wi = w[0:d]
    levy_sum = tf.reduce_sum(
      (wi - 1) ** 2 * (1 + 10 * tf.sin(pi * wi + 1) ** 2), axis=0)
    return term1 + levy_sum + term3


class F7(core.Function):
  """Griewank [Molga and Smutnicki 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    griewank_sum = tf.reduce_sum(x ** 2, axis=0) / 4000.0
    den = tf.range(1, x.shape[0] + 1, dtype=x.dtype)
    prod = tf.cos(x / tf.sqrt(den))
    prod = tf.reduce_prod(prod, axis=0)
    return griewank_sum - prod + 1


class F8(core.Function):
  """Rastrigin [Molga and Smutnicki 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.12, max=5.12)):
    super().__init__(domain)

  def __call__(self, x: tf.Tensor):
    if x.dtype != tf.float32:
      x = tf.cast(x, tf.float32)

    d = x.shape[0]
    return 10 * d + tf.reduce_sum(x ** 2 - 10 * tf.cos(x * 2 * np.math.pi),
                                  axis=0)


# Calcula os gradientes da função 'fun' na posição 'pos'.
# Só pode ser utilizada quando 'fun' é uma função implementada com
#   o TensorFlow.
def get_grads(fun: core.Function, pos: tf.Tensor):
  if pos.dtype != tf.float32:
    pos = tf.cast(pos, tf.float32)

  with tf.GradientTape() as tape:
    tape.watch(pos)
    y = fun(pos)

  return tape.gradient(y, pos), y
