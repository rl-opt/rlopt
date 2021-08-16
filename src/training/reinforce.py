"""REINFORCE para aprender um algoritmo de otimização."""

import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_episode_driver as dy_ed
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.networks import actor_distribution_network as actor_net
from tf_agents.networks import value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from src.environment import py_function_env as py_fun_env
from src.functions import numpy_functions as npf
from src.evaluation import utils as eval_utils
from src.training import utils as training_utils

if __name__ == '__main__':
  num_episodes = 2000  # Quantidade de episódios de treino.

  actor_lr = 3e-4  # Taxa de aprendizagem para o 'actor'.
  value_lr = 3e-4  # Taxa de aprendizagem para a 'value network'.
  discount = 0.99  # Fator de desconto.

  actor_layer_params = [256, 256]  # Camadas e unidades para a 'actor network'.
  value_layer_params = [256, 256]  # Camadas e unidades para a 'value network'.

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
    fc_layer_params=
    actor_layer_params)

  value_network = value_network.ValueNetwork(input_tensor_spec=obs_spec,
                                             fc_layer_params=value_layer_params)

  actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=actor_lr)
  value_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=value_lr)

  train_step = train_utils.create_train_step()

  agent = reinforce_agent.ReinforceAgent(time_step_spec=time_spec,
                                         action_spec=act_spec,
                                         actor_network=actor_network,
                                         value_network=value_network,
                                         optimizer=actor_optimizer,
                                         gamma=discount,
                                         normalize_returns=False,
                                         train_step_counter=train_step)

  agent.initialize()

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_training.batch_size,
    max_length=steps + 1)

  driver = dy_ed.DynamicEpisodeDriver(env=tf_env_training,
                                      policy=agent.collect_policy,
                                      observers=[replay_buffer.add_batch],
                                      num_episodes=1)

  driver.run = common.function(driver.run)

  agent.train = common.function(agent.train)
  agent.train_step_counter.assign(0)

  for ep in range(num_episodes):
    driver.run()
    experience = replay_buffer.gather_all()
    agent.train(experience)

    observations = tf.unstack(experience.observation[0])
    rewards = tf.unstack(experience.reward[0])
    best_solution = min([function(x.numpy()) for x in observations])
    ep_rew = sum(rewards)
    print(
      'episode = {0} '
      'Best solution on episode: {1} '
      'Return on episode: {2}'.format(ep, best_solution, ep_rew))

    # Limpando o replay buffer, só são utilizadas experiências do episódio
    #   atual.
    replay_buffer.clear()

  # Avaliação do algoritmo aprendido (policy) em 100 episódios distintos.
  # Produz um gráfico de convergência para o agente na função.
  eval_utils.evaluate_agent(tf_env_eval,
                            agent.policy,
                            function,
                            dims,
                            algorithm_name='REINFORCE',
                            save_to_file=False)

  # Salvamento da policy aprendida.
  # Pasta de saída: output/REINFORCE-{dims}D-{function.name}/
  # OBS:. Caso já exista, a saída é sobrescrita.
  training_utils.save_policy('REINFORCE',
                             function,
                             dims,
                             agent.policy)
