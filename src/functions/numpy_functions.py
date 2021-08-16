"""Implementação das diferentes funções de benchmark em NumPy."""

import numpy as np

import src.functions.core as core


class F1(core.Function):
  """Sphere (First function of De Jong) [Molga and Smutnicki 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.12, max=5.12)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    return np.sum(x * x, axis=0)


class F2(core.Function):
  """Rosenbrock (Second Function of De Jong) [Molga and Smutnicki 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    xi = x[:-1]
    xnext = x[1:]
    return np.sum(100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2, axis=0)


class F3(core.Function):
  """SumSquares (Axis Parallel Hyper-Ellipsoid) [Molga and Smutnicki 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    d = x.shape[0]
    mul = np.arange(start=1, stop=(d + 1), dtype=x.dtype)
    return np.sum((x ** 2) * mul, axis=0)


class F4(core.Function):
  """Rotated Hyper-Ellipsoid [Molga and Smutnicki 2005]."""

  def __init__(self,
               domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    d = x.shape[0]

    return np.sum([np.sum(x[0:(i + 1)] ** 2, axis=0) for i in range(d)],
                  dtype=np.float32, axis=0)


class F5(core.Function):
  """Ackley [Molga and Smutnicki 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-32.768, max=32.768),
               a=20, b=0.2, c=2 * np.math.pi):
    super().__init__(domain)
    self._a = a
    self._b = b
    self._c = c

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    d = x.shape[0]
    return -self.a * np.exp(-self.b * np.sqrt(np.sum(x * x, axis=0) / d)) - \
           np.exp(np.sum(np.cos(self.c * x), axis=0) / d) + self.a + np.math.e

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

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    pi = np.math.pi
    d = x.shape[0] - 1
    w = 1 + (x - 1) / 4

    term1 = np.sin(pi * w[0]) ** 2
    term3 = (w[d] - 1) ** 2 * (1 + np.sin(2 * pi * w[d]) ** 2)

    wi = w[0:d]
    levy_sum = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(pi * wi + 1) ** 2),
                      axis=0)
    return term1 + levy_sum + term3


class F7(core.Function):
  """Griewank [Molga and Smutnicki 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-10.0, max=10.0)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    griewank_sum = np.sum(x ** 2, axis=0) / 4000.0
    den = np.arange(start=1, stop=(x.shape[0] + 1), dtype=x.dtype)
    prod = np.cos(x / np.sqrt(den))
    prod = np.prod(prod, axis=0)
    return griewank_sum - prod + 1


class F8(core.Function):
  """Rastrigin [Molga and Smutnicki 2005]."""

  def __init__(self, domain: core.Domain = core.Domain(min=-5.12, max=5.12)):
    super().__init__(domain)

  def __call__(self, x: np.ndarray):
    if x.dtype != np.float32:
      x = x.astype(np.float32, casting='same_kind')

    d = x.shape[0]
    return 10 * d + np.sum(x ** 2 - 10 * np.cos(x * 2 * np.math.pi), axis=0)
