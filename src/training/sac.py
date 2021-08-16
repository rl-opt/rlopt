"""SAC para aprender um algoritmo de otimização."""

import numpy as np
import tensorflow as tf
from tf_agents.agents.ddpg import critic_network as critic_net
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network as tanh_net
from tf_agents.drivers import dynamic_step_driver as dy_sd
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.networks import actor_distribution_network as actor_net
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from src.environment import py_function_env as py_fun_env
from src.functions import numpy_functions as npf
from src.evaluation import utils as eval_utils
from src.training import utils as training_utils

if __name__ == '__main__':
  num_episodes = 2000  # Quantidade de episódios de treino.
  initial_collect_episodes = 20  # Quantidade de episódios de coleta inicial.
  c_steps_per_it = 1  # Quantidade de passos por iteração.

  replay_buffer_capacity = 1000000  # Capacidade do replay buffer.
  batch_size = 256  # Tamanho do batch.

  actor_lr = 3e-4  # Taxa de aprendizagem para o 'actor'.
  critic_lr = 3e-4  # Taxa de aprendizagem para o 'critic'.
  target_update_tau = 5e-3  # Valor para o 'tau'.
  alpha_lr = 3e-4
  target_update_period = 2

  discount = 0.99  # Fator de desconto.

  actor_layer_params = [256, 256]  # Camadas e unidades para a 'actor network'.
  critic_observation_layer_params = None
  critic_action_layer_params = None
  # Camadas e unidades para a 'critic network'.
  critic_joint_layer_params = [256, 256]

  steps = 250  # Quantidade de interações agente-ambiente para treino.
  steps_eval = 500  # Quantidade de interações agente-ambiente para avaliação.

  dims = 30  # Dimensões da função.
  function = npf.F1()  # Função (F1--F8).

  env_training = py_fun_env.PyFunctionEnvironment(function=function,
                                                  dims=dims)
  env_training = wrappers.TimeLimit(env=env_training, duration=steps)

  env_eval = py_fun_env.PyFunctionEnvironment(function=function,
                                              dims=dims)
  env_eval = wrappers.TimeLimit(env=env_eval, duration=steps)

  tf_env_training = tf_py_environment.TFPyEnvironment(environment=env_training)
  tf_env_eval = tf_py_environment.TFPyEnvironment(environment=env_eval)

  obs_spec = tf_env_training.observation_spec()
  act_spec = tf_env_training.action_spec()
  time_spec = tf_env_training.time_step_spec()

  actor_network = actor_net.ActorDistributionNetwork(
    input_tensor_spec=obs_spec,
    output_tensor_spec=act_spec,
    fc_layer_params=actor_layer_params,
    continuous_projection_net=tanh_net.TanhNormalProjectionNetwork)

  critic_network = critic_net.CriticNetwork(
    input_tensor_spec=(obs_spec, act_spec),
    observation_fc_layer_params=
    critic_observation_layer_params,
    action_fc_layer_params=critic_action_layer_params,
    joint_fc_layer_params=critic_joint_layer_params,
    activation_fn=tf.keras.activations.relu,
    output_activation_fn=tf.keras.activations.linear,
    kernel_initializer='glorot_uniform',
    last_kernel_initializer='glorot_uniform')

  actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
  critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
  alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)

  train_step = train_utils.create_train_step()

  agent = sac_agent.SacAgent(time_step_spec=time_spec,
                             action_spec=act_spec,
                             critic_network=critic_network,
                             critic_optimizer=critic_optimizer,
                             actor_network=actor_network,
                             actor_optimizer=actor_optimizer,
                             alpha_optimizer=alpha_optimizer,
                             gamma=discount,
                             target_update_tau=target_update_tau,
                             target_update_period=target_update_period,
                             td_errors_loss_fn=tf.math.squared_difference,
                             train_step_counter=train_step)

  agent.initialize()

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_training.batch_size,
    max_length=replay_buffer_capacity)

  dataset = replay_buffer.as_dataset(
    sample_batch_size=batch_size,
    num_steps=2).prefetch(64)

  iterator = iter(dataset)

  initial_driver = dy_sd.DynamicStepDriver(env=tf_env_training,
                                           policy=agent.collect_policy,
                                           observers=[replay_buffer.add_batch],
                                           num_steps=c_steps_per_it)

  driver = dy_sd.DynamicStepDriver(env=tf_env_training,
                                   policy=agent.collect_policy,
                                   observers=[replay_buffer.add_batch],
                                   num_steps=c_steps_per_it)

  driver.run = common.function(driver.run)
  initial_driver.run = common.function(initial_driver.run)

  for _ in range(initial_collect_episodes):
    done = False
    while not done:
      time_step, _ = initial_driver.run()
      done = time_step.is_last()

  agent.train = common.function(agent.train)
  agent.train_step_counter.assign(0)

  for ep in range(num_episodes):
    done = False
    best_solution = np.finfo(np.float32).max
    ep_rew = 0.0
    while not done:
      time_step, _ = driver.run()
      experience, unused_info = next(iterator)
      agent.train(experience)

      obj_value = driver.env.get_info().objective_value[0]

      if obj_value < best_solution:
        best_solution = obj_value

      ep_rew += time_step.reward
      done = time_step.is_last()

    print('episode = {0} '
          'Best solution on episode: {1} '
          'Return on episode: {2}'.format(ep, best_solution, ep_rew))

  # Avaliação do algoritmo aprendido (policy) em 100 episódios distintos.
  # Produz um gráfico de convergência para o agente na função.
  eval_utils.evaluate_agent(tf_env_eval,
                            agent.policy,
                            function,
                            dims,
                            algorithm_name='SAC',
                            save_to_file=False)

  # Salvamento da policy aprendida.
  # Pasta de saída: output/SAC-{dims}D-{function.name}/
  # OBS:. Caso já exista, a saída é sobrescrita.
  training_utils.save_policy('SAC',
                             function,
                             dims,
                             agent.policy)
