"""TD3 para aprender um algoritmo de otimização."""

import numpy as np
import tensorflow as tf
from tf_agents.agents.td3 import td3_agent
from tf_agents.agents.ddpg import actor_network as actor_net
from tf_agents.agents.ddpg import critic_network as critic_net
from tf_agents.drivers import dynamic_step_driver as dy_sd
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
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
  collect_steps_per_iteration = 1  # Quantidade de passos por iteração.

  buffer_size = 1000000  # Capacidade do replay buffer.
  batch_size = 256  # Tamanho do batch.

  actor_lr = 3e-4  # Taxa de aprendizagem para o 'actor'.
  critic_lr = 3e-4  # Taxa de aprendizagem para o 'critic'.
  tau = 5e-3  # Valor para o 'tau'.
  actor_update_period = 2
  target_update_period = 2

  discount = 0.99  # Fator de desconto.

  exploration_noise_std = 0.1
  target_policy_noise = 0.2
  target_policy_noise_clip = 0.5

  actor_layer_params = [256, 256]  # Camadas e unidades para a 'actor network'.

  critic_action_fc_layer_params = None
  critic_observation_fc_layer_params = None
  # Camadas e unidades para a 'critic network'.
  critic_fc_layer_params = [256, 256]

  steps = 250  # Quantidade de interações agente-ambiente para treino.
  steps_eval = 500  # Quantidade de interações agente-ambiente para avaliação.

  dims = 30  # Dimensões da função.
  function = npf.F1()  # Função (F1--F8).

  env_training = py_fun_env.PyFunctionEnvironment(function=function,
                                                  dims=dims)
  env_training = wrappers.TimeLimit(env=env_training, duration=steps)

  env_eval = py_fun_env.PyFunctionEnvironment(function=function,
                                              dims=dims)
  env_eval = wrappers.TimeLimit(env=env_eval, duration=steps_eval)

  tf_env_training = tf_py_environment.TFPyEnvironment(environment=env_training)
  tf_env_eval = tf_py_environment.TFPyEnvironment(environment=env_eval)

  obs_spec = tf_env_training.observation_spec()
  act_spec = tf_env_training.action_spec()
  time_spec = tf_env_training.time_step_spec()

  actor_network = actor_net.ActorNetwork(
    input_tensor_spec=obs_spec,
    output_tensor_spec=act_spec,
    fc_layer_params=actor_layer_params,
    activation_fn=tf.keras.activations.relu)
  critic_network = critic_net.CriticNetwork(
    input_tensor_spec=(obs_spec, act_spec),
    observation_fc_layer_params=critic_observation_fc_layer_params,
    action_fc_layer_params=critic_action_fc_layer_params,
    joint_fc_layer_params=critic_fc_layer_params,
    activation_fn=tf.keras.activations.relu,
    output_activation_fn=tf.keras.activations.linear)

  actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
  critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

  train_step = train_utils.create_train_step()

  agent = td3_agent.Td3Agent(
    time_step_spec=time_spec,
    action_spec=act_spec,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    target_update_tau=tau,
    exploration_noise_std=exploration_noise_std,
    target_policy_noise=target_policy_noise,
    target_policy_noise_clip=target_policy_noise_clip,
    actor_update_period=actor_update_period,
    target_update_period=target_update_period,
    train_step_counter=train_step,
    gamma=discount)

  agent.initialize()

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_training.batch_size,
    max_length=buffer_size)

  driver = dy_sd.DynamicStepDriver(env=tf_env_training,
                                   policy=agent.collect_policy,
                                   observers=[replay_buffer.add_batch],
                                   num_steps=collect_steps_per_iteration)

  initial_collect_driver = dy_sd.DynamicStepDriver(
    env=tf_env_training,
    policy=agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)

  initial_collect_driver.run = common.function(initial_collect_driver.run)
  driver.run = common.function(driver.run)
  agent.train = common.function(agent.train)

  for _ in range(initial_collect_episodes):
    done = False
    while not done:
      time_step, _ = initial_collect_driver.run()
      done = time_step.is_last()

  dataset = replay_buffer.as_dataset(
    sample_batch_size=batch_size,
    num_steps=2).prefetch(64)

  iterator = iter(dataset)

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
                            algorithm_name='TD3',
                            save_to_file=False)

  # Salvamento da policy aprendida.
  # Pasta de saída: output/TD3-{dims}D-{function.name}/
  # OBS:. Caso já exista, a saída é sobrescrita.
  training_utils.save_policy('TD3',
                             function,
                             dims,
                             agent.policy)
