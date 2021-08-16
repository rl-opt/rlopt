"""Calcula as medidas estatísticas (média, desvio padrão) dos algoritmos aprendidos."""

import csv
import os
from typing import List, NamedTuple

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers

from src.environment import py_function_env as py_fun_env
from src.functions import numpy_functions as npf
from src.functions import core
from src.evaluation import utils as eval_utils
from src import config

MODELS_DIR = config.POLICIES_DIR


# Agrupa uma policy e uma função.
class PolicyAndFunction(NamedTuple):
  policy: object
  function: core.Function


# Agrupa as informações necessárias sobre a avaliação da policy.
class PolicyEvalData(NamedTuple):
  policy_function: PolicyAndFunction
  avg_best_solution: float
  stddev_best_solutions: float
  avg_best_solution_iteration: int


def load_policies(functions: List[core.Function],
                  dims: int,
                  algorithm: str) -> List[PolicyAndFunction]:
  models_dir = os.path.join(MODELS_DIR, f'{algorithm}')
  models_dir = os.path.join(models_dir, f'{str(dims)}D')

  pairs: [PolicyAndFunction] = []

  for fun in functions:
    policy_dir = os.path.join(models_dir, fun.name)
    if os.path.exists(policy_dir):
      policy = tf.compat.v2.saved_model.load(policy_dir)
      pairs.append(PolicyAndFunction(policy, fun))
    else:
      print('{0} não foi incluído na lista.'.format(fun.name))

  return pairs


def evaluate_policies(policies_functions: List[PolicyAndFunction],
                      dims: int,
                      steps=500,
                      episodes=100) -> List[PolicyEvalData]:
  # Calcula as medidas estáticas para uma única 'PolicyAndFunction'.
  def evaluate_policy(policy_function: PolicyAndFunction) -> PolicyEvalData:
    nonlocal steps
    nonlocal episodes
    nonlocal dims

    env = py_fun_env.PyFunctionEnvironment(
      function=policy_function.function,
      dims=dims)
    env = wrappers.TimeLimit(env=env, duration=steps)
    tf_env = tf_py_environment.TFPyEnvironment(environment=env)

    policy = policy_function.policy

    best_solutions: [np.float32] = []
    best_solutions_iterations: [int] = []

    for _ in range(episodes):
      time_step = tf_env.reset()
      info = tf_env.get_info()
      it = 0

      best_solution_ep = info.objective_value[0]
      best_it_ep = it

      while not time_step.is_last():
        it += 1
        action_step = policy.action(time_step)
        time_step = tf_env.step(action_step.action)
        info = tf_env.get_info()

        obj_value = info.objective_value[0]

        if obj_value < best_solution_ep:
          best_solution_ep = obj_value
          best_it_ep = it

      best_solutions.append(best_solution_ep)
      best_solutions_iterations.append(best_it_ep)

    avg_best_solution = np.mean(best_solutions).astype(np.float32)
    avg_best_solution_time = np.rint(np.mean(best_solutions_iterations))
    stddev_best_solutions = np.std(best_solutions).astype(np.float32)

    return PolicyEvalData(
      policy_function=policy_function,
      avg_best_solution=avg_best_solution.item(),
      stddev_best_solutions=stddev_best_solutions.item(),
      avg_best_solution_iteration=int(avg_best_solution_time))

  policies_evaluation_data = [evaluate_policy(p) for p in policies_functions]

  return policies_evaluation_data


def write_to_csv(policies_evaluation_data: List[PolicyEvalData],
                 dims: int,
                 file_name: str):
  with open(file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Function',
                     'Dims',
                     'Avg Best Solution',
                     'Stddev of best solutions',
                     'Avg Best Solution Iteration'])
    for data in policies_evaluation_data:
      writer.writerow([data.policy_function.function.name,
                       dims,
                       data.avg_best_solution,
                       data.stddev_best_solutions,
                       data.avg_best_solution_iteration])


if __name__ == "__main__":
  # Dimensões das funções.
  DIMS = 30
  # Quantidade de episódios para o cálculo das medidas.
  EPISODES = 100
  # Quantidade de iterações.
  STEPS = 500
  # Lista com as funções que serão testadas.
  FUNCTIONS = [npf.F1(), npf.F2(), npf.F3(), npf.F4(),
               npf.F5(), npf.F6(), npf.F7(), npf.F8()]
  # Nome do algoritmo de otimização a ser utilizado:
  # 'PPO', 'REINFORCE', 'SAC' ou 'TD3'
  ALGORITHM = 'PPO'

  # Carrega as policies (algoritmos de otimização) para cada uma das
  #   funções 'FUNCTION' em 'FUNCTIONS'.
  # As policies carregadas se encontram em:
  #   'policies/{ALGORITHM}/{DIMS}D/{FUNCTION.name}'.
  policies_and_functions = load_policies(FUNCTIONS,
                                         DIMS,
                                         ALGORITHM)

  # Calcula as medidas estatísticas
  eval_data = evaluate_policies(policies_and_functions,
                                DIMS,
                                episodes=EPISODES,
                                steps=STEPS)

  # Salva os resultados em um csv.
  write_to_csv(eval_data,
               DIMS,
               file_name=f'{ALGORITHM}_{DIMS}D_data.csv')

  # Realiza o teste com os baselines.
  """
  eval_utils.evaluate_baselines(functions=FUNCTIONS,
                                dims=DIMS,
                                steps=STEPS,
                                episodes=EPISODES)
  """
