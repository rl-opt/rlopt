"""Funções utilitárias para avaliação dos algoritmos."""

import typing
import csv

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import pandas as pd
import tensorflow as tf
from tf_agents.environments import tf_environment
from tf_agents.policies import tf_policy

from src.functions import core
from src.functions import tensorflow_functions as tff


# Agrupa as informações necessárias sobre a avaliação de um baseline.
class BaselineEvalData(typing.NamedTuple):
  baseline_name: str
  function_name: str
  avg_best_solution: float
  stddev_best_solutions: float
  avg_best_solution_iteration: int


def evaluate_agent(eval_env: tf_environment.TFEnvironment,
                   policy_eval: tf_policy.TFPolicy,
                   function: core.Function,
                   dims,
                   algorithm_name,
                   save_to_file=False,
                   show_all_trajectories=False,
                   episodes=100):
  trajectories = []
  best_trajectory = [np.finfo(np.float32).max]
  best_pos = None

  for _ in range(episodes):
    time_step = eval_env.reset()
    info = eval_env.get_info()

    best_solution_ep = info.objective_value[0]
    best_pos_ep = info.position[0]

    trajectory = [best_solution_ep]
    it = 0

    while not time_step.is_last():
      it += 1
      action_step = policy_eval.action(time_step)
      time_step = eval_env.step(action_step.action)
      info = eval_env.get_info()

      obj_value = info.objective_value[0]

      if obj_value < best_solution_ep:
        best_solution_ep = obj_value
        best_pos_ep = info.position[0]

      trajectory.append(best_solution_ep)

    if trajectory[-1] < best_trajectory[-1]:
      best_trajectory = trajectory
      best_pos = best_pos_ep

    trajectories.append(trajectory)

  mean = np.mean(trajectories, axis=0)

  _, ax = plt.subplots(figsize=(18.0, 10.0,))

  if show_all_trajectories:
    for t in trajectories:
      ax.plot(t, '--c', alpha=0.4)

  ax.plot(mean, 'r', label='Best mean value: {0}'.format(mean[-1]))
  ax.plot(best_trajectory, 'g',
          label='Best value: {0}'.format(best_trajectory[-1]))

  ax.set(xlabel="Iterations\nBest solution at: {0}".format(best_pos),
         ylabel="Best objective value",
         title="{0} on {1} ({2} Dims)".format(algorithm_name,
                                              function.name,
                                              dims))

  ax.set_yscale('log')
  ax.set_xlim(left=0)

  ax.legend()
  ax.grid()

  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")
  if save_to_file:
    plt.savefig(fname='{0}-{1}dims-{2}.png'.format(function.name,
                                                   dims,
                                                   algorithm_name),
                bbox_inches='tight')
  plt.show()


def evaluate_baselines(functions: typing.List[core.Function],
                       dims: int,
                       steps=500,
                       episodes=100):
  baseline_eval_data: typing.List[BaselineEvalData] = []

  for fun in functions:
    rng = rand.default_rng()
    tf_function = get_tf_function(fun)

    gd_bs = []
    gd_bs_it = []

    nag_bs = []
    nag_bs_it = []

    for ep in range(episodes):
      gd_pos = rng.uniform(size=(dims,),
                           low=fun.domain.min,
                           high=fun.domain.max)

      nag_pos = rng.uniform(size=(dims,),
                            low=fun.domain.min,
                            high=fun.domain.max)

      gd = GD(tf_function,
              pos=gd_pos,
              steps=steps)
      gd_bs.append(gd[0][-1])
      gd_bs_it.append(gd[1])

      nag = NAG(tf_function,
                pos=nag_pos,
                steps=steps)
      nag_bs.append(nag[0][-1])
      nag_bs_it.append(nag[1])

    data_gd = BaselineEvalData('GD',
                               tf_function.name,
                               np.mean(gd_bs).astype(np.float32).item(),
                               np.std(gd_bs).astype(np.float32).item(),
                               int(np.rint(np.mean(gd_bs_it))))
    baseline_eval_data.append(data_gd)

    data_nag = BaselineEvalData('NAG',
                                tf_function.name,
                                np.mean(nag_bs).astype(np.float32).item(),
                                np.std(nag_bs).astype(np.float32).item(),
                                int(np.rint(np.mean(nag_bs_it))))
    baseline_eval_data.append(data_nag)

  with open(f'baselines_{dims}D_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Baseline',
                     'Function',
                     'Dims',
                     'Avg Best Solution',
                     'Stddev of best solutions',
                     'Avg Best Solution Iteration'])
    for data in baseline_eval_data:
      writer.writerow([data.baseline_name,
                       data.function_name,
                       dims,
                       data.avg_best_solution,
                       data.stddev_best_solutions,
                       data.avg_best_solution_iteration])


# Função utilitária para obter uma função equivalente de Numpy em Tensorflow.
def get_tf_function(function: core.Function):
  domain = function.domain
  f = getattr(tff, function.name)
  return f(domain)


# Parâmetros para o GD em cada função.
# Valores escolhidos após realizar 20 testes distintos (500 iterações)
#   e selecionando o parâmetro que obteve a melhor solução final (média).
# Conjunto de busca: {1e-1, 1e-2, 1e-3, 1e-4}
gd_lrs = {'F1': 1e-1,
          'F2': 1e-2,
          'F3': 1e-4,
          'F4': 1e-4,
          'F5': 1e-2,
          'F6': 1e-1,
          'F7': 1e-1,
          'F8': 1e-4}


# Baseline: GD (Gradient Descent)
def GD(function: core.Function,
       pos: typing.Union[tf.Tensor, np.ndarray],
       steps=500):
  lr = gd_lrs.get(function.name, 1e-2)
  pos = tf.convert_to_tensor(pos, dtype=tf.float32)
  best_solutions = [function(pos)]
  best_it = 0
  domain = function.domain

  for t in range(steps):
    grads, _ = tff.get_grads(function, pos)
    pos = tf.clip_by_value(pos - lr * grads,
                           clip_value_min=domain.min,
                           clip_value_max=domain.max)

    y = function(pos)
    if y < best_solutions[-1]:
      best_it = t
    best_solutions.append(min(y, best_solutions[-1]))
  return best_solutions, best_it


# Parâmetros para o NAG em cada função.
# Valores escolhidos após realizar 20 testes distintos e selecionando o par
#   (lr, momentum) de parâmetros que encontraram a melhor solução final (média)
# Conjunto de busca: {1e-1, 1e-2, 1e-3, 1e-4} X {0.5, 0.8, 0.9}
nag_params = {'F1': (1e-1, 0.5),
              'F2': (1e-3, 0.5),
              'F3': (1e-4, 0.9),
              'F4': (1e-4, 0.9),
              'F5': (1e-1, 0.8),
              'F6': (1e-3, 0.9),
              'F7': (1e-1, 0.9),
              'F8': (1e-4, 0.9)}


# Baseline: NAG (Nesterov accelerated gradient)
def NAG(function: core.Function,
        pos: typing.Union[tf.Tensor, np.ndarray],
        steps=500):
  lr, momentum = nag_params.get(function.name, (1e-2, 0.8))
  pos = tf.convert_to_tensor(pos, dtype=tf.float32)
  velocity = tf.zeros(shape=pos.shape, dtype=tf.float32)
  domain = function.domain

  best_solutions = [function(pos)]
  best_it = 0

  for t in range(steps):
    projected = tf.clip_by_value(pos + momentum * velocity,
                                 clip_value_min=domain.min,
                                 clip_value_max=domain.max)
    grads, _ = tff.get_grads(function, projected)

    velocity = momentum * velocity - lr * grads
    pos = tf.clip_by_value(pos + velocity,
                           clip_value_min=domain.min,
                           clip_value_max=domain.max)

    current_pos = function(pos)
    if current_pos < best_solutions[-1]:
      best_it = t
    best_solutions.append(min(current_pos, best_solutions[-1]))
  return best_solutions, best_it


def plot_convergence(show=False, dpi=300, style=None):
  if style is None:
    style = ['-', '-', '-', '-', '--', '--']

  for i in range(1, 9):
    F = f'F{i}'
    file = F + '_30D_convergence.csv'
    data = pd.read_csv(file)
    del data['iteration']
    data.plot(grid=True, fontsize=9, style=style)
    plt.xlabel('Iterações')
    plt.ylabel('Melhor Valor')
    plt.savefig(F + '_30D_plot', dpi=dpi)
  if show:
    plt.show()
