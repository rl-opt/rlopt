"""Realiza a comparação da convergência com os diferentes algoritmos."""

import os
from typing import List, NamedTuple

import numpy as np
import tensorflow as tf
import pandas as pd

from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.policies import tf_policy

from src.environment import py_function_env as py_fun_env
from src.functions import core
from src.functions import numpy_functions as npf
from src import config
from src.evaluation import utils as eval_utils

MODELS_DIR = config.POLICIES_DIR


class Trajectory(NamedTuple):
  list_best_values: np.ndarray
  name: str


def write_to_csv(trajectories: List[Trajectory],
                 function: core.Function,
                 dims: int):
  file_name = f'{function.name}_{dims}D_convergence.csv'
  data = pd.DataFrame({t.name: t.list_best_values for t in trajectories})
  data.to_csv(file_name, index_label='iteration')


def run_episode(tf_eval_env: tf_environment.TFEnvironment,
                policy: tf_policy.TFPolicy,
                trajectory_name: str,
                function: core.Function) -> Trajectory:
  time_step = tf_eval_env.current_time_step()

  best_pos = time_step.observation.numpy()[0]
  best_solution = function(best_pos)
  best_values_at_it = [best_solution]

  it = 0

  while True:
    it += 1
    action_step = policy.action(time_step)
    time_step = tf_eval_env.step(action_step.action)

    pos = time_step.observation.numpy()[0]
    obj_value = function(pos)

    if obj_value < best_solution:
      best_solution = obj_value

    if time_step.is_last():
      break
    best_values_at_it.append(best_solution)

  return Trajectory(list_best_values=np.array(best_values_at_it,
                                              dtype=np.float32),
                    name=trajectory_name)


def run_rl_agent(policy: tf_policy.TFPolicy,
                 trajectory_name: str,
                 num_steps: int,
                 function: core.Function,
                 dims: int,
                 initial_time_step) -> Trajectory:
  env = py_fun_env.PyFunctionEnvironment(function, dims)
  env = wrappers.TimeLimit(env, duration=num_steps)

  tf_eval_env = tf_py_environment.TFPyEnvironment(environment=env)
  tf_eval_env._time_step = initial_time_step

  return run_episode(tf_eval_env=tf_eval_env,
                     policy=policy,
                     trajectory_name=trajectory_name,
                     function=function)


def get_average_trajectory(trajectories: List[Trajectory]):
  best_values = []
  name = trajectories[0].name

  for t in trajectories:
    best_values.append(t.list_best_values)

  return Trajectory(
    list_best_values=np.mean(np.array(best_values, dtype=np.float32), axis=0),
    name=name)


def policy_path(agent: str, dims: int, fun: core.Function):
  return os.path.join(MODELS_DIR, agent, str(dims) + 'D', fun.name)


if __name__ == '__main__':
  # Dimensões das funções.
  DIMS = 30
  # Quantidade de episódios para o cálculo das medidas.
  EPISODES = 100
  # Quantidade de interações agente-ambiente.
  STEPS = 500
  # Lista com as funções que serão testadas.
  FUNCTIONS = [npf.F1(), npf.F2(), npf.F3(), npf.F4(),
               npf.F5(), npf.F6(), npf.F7(), npf.F8()]

  for FUNCTION in FUNCTIONS:
    ENV = py_fun_env.PyFunctionEnvironment(FUNCTION, DIMS)
    ENV = wrappers.TimeLimit(ENV, duration=STEPS)
    TF_ENV = tf_py_environment.TFPyEnvironment(environment=ENV)
    TF_FUNCTION = eval_utils.get_tf_function(FUNCTION)

    reinforce_policy = tf.compat.v2.saved_model.load(policy_path('REINFORCE',
                                                                 DIMS,
                                                                 FUNCTION))
    reinforce_trajectories: List[Trajectory] = []

    sac_policy = tf.compat.v2.saved_model.load(policy_path('SAC',
                                                           DIMS,
                                                           FUNCTION))
    sac_trajectories: List[Trajectory] = []

    td3_policy = tf.compat.v2.saved_model.load(policy_path('TD3',
                                                           DIMS,
                                                           FUNCTION))
    td3_trajectories: List[Trajectory] = []

    ppo_policy = tf.compat.v2.saved_model.load(policy_path('PPO',
                                                           DIMS,
                                                           FUNCTION))
    ppo_trajectories: List[Trajectory] = []

    gd_trajectories: List[Trajectory] = []

    nag_trajectories: List[Trajectory] = []

    for _ in range(EPISODES):
      initial_ts = TF_ENV.reset()

      reinforce_trajectories.append(run_rl_agent(policy=reinforce_policy,
                                                 trajectory_name='REINFORCE',
                                                 num_steps=STEPS,
                                                 function=FUNCTION,
                                                 dims=DIMS,
                                                 initial_time_step=initial_ts))

      sac_trajectories.append(run_rl_agent(policy=sac_policy,
                                           trajectory_name='SAC',
                                           num_steps=STEPS,
                                           function=FUNCTION,
                                           dims=DIMS,
                                           initial_time_step=initial_ts))

      td3_trajectories.append(run_rl_agent(policy=td3_policy,
                                           trajectory_name='TD3',
                                           num_steps=STEPS,
                                           function=FUNCTION,
                                           dims=DIMS,
                                           initial_time_step=initial_ts))

      ppo_trajectories.append(run_rl_agent(policy=ppo_policy,
                                           trajectory_name='PPO',
                                           num_steps=STEPS,
                                           function=FUNCTION,
                                           dims=DIMS,
                                           initial_time_step=initial_ts))

      gd_list = eval_utils.GD(function=TF_FUNCTION,
                              pos=initial_ts.observation[0],
                              steps=STEPS)[0]
      gd_trajectories.append(Trajectory(gd_list, 'GD'))

      nag_list = eval_utils.NAG(function=TF_FUNCTION,
                                pos=initial_ts.observation[0],
                                steps=STEPS)[0]
      nag_trajectories.append(Trajectory(nag_list, 'NAG'))

    avg_reinforce = get_average_trajectory(reinforce_trajectories)
    avg_sac = get_average_trajectory(sac_trajectories)
    avg_td3 = get_average_trajectory(td3_trajectories)
    avg_ppo = get_average_trajectory(ppo_trajectories)
    avg_gd = get_average_trajectory(gd_trajectories)
    avg_nag = get_average_trajectory(nag_trajectories)

    write_to_csv([avg_reinforce, avg_sac, avg_td3, avg_ppo, avg_gd, avg_nag],
                 function=FUNCTION,
                 dims=DIMS)
