"""Ambiente utilizado para a minimização de funções matemáticas com RL."""

import collections

import numpy as np
from numpy.random import default_rng
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from src.functions import core

# Quantidade máxima de iterações entre o agente e ambiente.
MAX_STEPS = 50000


class FunctionEnvironmentInfo(
  collections.namedtuple('FunctionEnvironmentInfo',
                         ('position', 'objective_value'))):
  """Representa algumas informações úteis sobre o ambiente.
  'position' representa a última posição do agente na função (x).
  'objective_value' representa o valor da função nessa posição (f(x)).
  """
  pass


class PyFunctionEnvironment(py_environment.PyEnvironment):
  """Ambiente para a minização de função.
  Dada uma função f: D -> I, onde D é um subonjunto de R^d
  e I é um subconjunto de R, as especificações do ambiente são:
    as observações (s em D) são posições do domínio;
    as ações (a em R^d) são os possíveis passos;
    as recompensas são r = -f(s + a).
  """
  def __init__(self, function: core.Function, dims):
    super().__init__()
    self._rng = default_rng()
    self.func = function
    self._dims = dims

    self._episode_ended = False
    self._steps_taken = 0

    self._state = self.__initial_state()

    self._last_objective_value = self.func(self._state)
    self._last_position = self._state

    self._action_spec = array_spec.BoundedArraySpec(shape=(self._dims,),
                                                    dtype=np.float32,
                                                    minimum=-1.0,
                                                    maximum=1.0,
                                                    name='action')

    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(self._dims,),
      dtype=np.float32,
      minimum=function.domain.min,
      maximum=function.domain.max,
      name='observation')

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def get_info(self):
    return FunctionEnvironmentInfo(position=self._last_position,
                                   objective_value=self._last_objective_value)

  def get_state(self):
    state = (self._state, self._steps_taken, self._episode_ended)
    return state

  def set_state(self, state):
    _state, _steps_taken, _episode_ended = state
    self._state = _state
    self._steps_taken = _steps_taken
    self._episode_ended = _episode_ended

  def _step(self, action):
    if self._episode_ended:
      return self.reset()

    self._state = self._state + action
    domain_min, domain_max = self.func.domain
    self._state = np.clip(self._state, domain_min, domain_max)

    self._steps_taken += 1
    if self._steps_taken > MAX_STEPS:
      self._episode_ended = True

    obj_value = self.func(self._state)
    reward = -obj_value
    self._last_objective_value = obj_value
    self._last_position = self._state

    if self._episode_ended:
      return ts.termination(self._state, reward)

    return ts.transition(self._state, reward)

  def _reset(self):
    self._state = self.__initial_state()
    self._episode_ended = False
    self._steps_taken = 0
    self._last_objective_value = self.func(self._state)
    self._last_position = self._state
    return ts.restart(self._state)

  def render(self, mode: str = 'human'):
    raise NotImplementedError("Not Implemented yet.")

  def __initial_state(self) -> np.ndarray:
    domain_min, domain_max = self.func.domain
    state = self._rng.uniform(size=(self._dims,),
                              low=domain_min,
                              high=domain_max)
    return state.astype(dtype=np.float32, copy=False)
