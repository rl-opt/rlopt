"""PPO para aprender um algoritmo de otimização."""

import tensorflow as tf
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_episode_driver as dy_ed
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from src.environment import py_function_env as py_func_env
from src.functions import numpy_functions as npf
from src.evaluation import utils as eval_utils
from src.training import utils as training_utils


def main(_):
  num_episodes = 2000  # Quantidade de episódios de treino.
  replay_buffer_capacity = 251  # Tamanho do replay buffer (por ambiente).
  num_epochs = 25  # Quantidade de épocas.
  num_parallel_environments = 15  # Quantidade de ambientes paralelos.

  learning_rate = 3e-4  # Taxa de aprendizagem.
  actor_fc_layers = (256, 256)  # Camadas e unidades para a 'actor network'.
  value_fc_layers = (256, 256)  # Camadas e unidades para a 'value network'.

  num_eval_episodes = 25  # Quantidade de episódios de avaliação.
  eval_interval = 100  # Intervalo de avaliação.
  steps = 250  # Quantidade de interações agente-ambiente para treino.
  steps_eval = 500  # Quantidade de interações agente-ambiente para avaliação.

  dims = 30  # Dimensões da função.
  function = npf.F1()  # Função (F1--F8).

  tf_env_eval = tf_py_environment.TFPyEnvironment(
    wrappers.TimeLimit(env=py_func_env.PyFunctionEnvironment(function=function,
                                                             dims=dims),
                       duration=steps_eval))

  def evaluate_current_policy(policy, episodes=num_eval_episodes):
    total_return = tf.Variable([0.])
    best_solutions = tf.Variable([0.])

    obj_value = tf.Variable([tf.float32.max])
    best_solution = tf.Variable([tf.float32.max])
    episode_return = tf.Variable([0.])

    def run_policy():
      for _ in range(episodes):
        time_step = tf_env_eval.reset()
        episode_return.assign([0.])
        best_solution.assign([tf.float32.max])

        while not time_step.is_last():
          action_step = policy.action(time_step)
          time_step = tf_env_eval.step(action_step.action)

          episode_return.assign_add(time_step.reward)
          obj_value.assign(tf_env_eval.get_info().objective_value)

          if tf.math.less(obj_value, best_solution):
            best_solution.assign(obj_value)

        best_solutions.assign_add(best_solution)
        total_return.assign_add(episode_return)

    run_policy()
    avg_return_ = tf.math.divide(total_return, episodes)
    avg_best_solution = tf.math.divide(best_solutions, episodes)
    return avg_best_solution, avg_return_

  global_step = train_utils.create_train_step()

  tf_env_training = tf_py_environment.TFPyEnvironment(
    parallel_py_environment.ParallelPyEnvironment(
      [lambda: wrappers.TimeLimit(
        env=py_func_env.PyFunctionEnvironment(function=function, dims=dims),
        duration=steps)] * num_parallel_environments))

  actor_net = actor_distribution_network.ActorDistributionNetwork(
    tf_env_training.observation_spec(),
    tf_env_training.action_spec(),
    fc_layer_params=actor_fc_layers)
  value_net = value_network.ValueNetwork(tf_env_training.observation_spec(),
                                         fc_layer_params=value_fc_layers,
                                         activation_fn=tf.keras.activations.
                                         relu)

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  agent = ppo_clip_agent.PPOClipAgent(
    tf_env_training.time_step_spec(),
    tf_env_training.action_spec(),
    optimizer,
    actor_net=actor_net,
    value_net=value_net,
    entropy_regularization=0.0,
    importance_ratio_clipping=0.2,
    normalize_observations=False,
    normalize_rewards=False,
    use_gae=True,
    num_epochs=num_epochs,
    train_step_counter=global_step)

  agent.initialize()

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    agent.collect_data_spec,
    batch_size=num_parallel_environments,
    max_length=replay_buffer_capacity)

  collect_driver = dy_ed.DynamicEpisodeDriver(tf_env_training,
                                              agent.collect_policy,
                                              observers=[
                                                replay_buffer.add_batch],
                                              num_episodes=1)

  collect_driver.run = common.function(collect_driver.run, autograph=False)
  agent.train = common.function(agent.train, autograph=False)

  for ep in range(num_episodes):
    collect_driver.run()

    experience = replay_buffer.gather_all()
    agent.train(experience=experience)
    # Limpando o replay buffer, só são utilizadas experiências do episódio
    #   atual.
    replay_buffer.clear()

    print('episode {0}'.format(ep))
    if ep % eval_interval == 0:
      avg_best_sol, avg_return = evaluate_current_policy(policy=agent.policy)
      print(
        'avg_best_solution: {0} avg_return: {1}'.format(avg_best_sol.numpy()[0],
                                                        avg_return.numpy()[0]))

  print('---- Training finished ----')
  avg_best_sol, avg_return = evaluate_current_policy(policy=agent.policy,
                                                     episodes=100)
  print('avg_best_solution: {0} avg_return: {1}'.format(avg_best_sol.numpy()[0],
                                                        avg_return.numpy()[0]))

  # Avaliação do algoritmo aprendido (policy) em 100 episódios distintos.
  # Produz um gráfico de convergência para o agente na função.
  eval_utils.evaluate_agent(tf_env_eval,
                            agent.policy,
                            function,
                            dims,
                            algorithm_name='PPO',
                            save_to_file=False)

  # Salvamento da policy aprendida.
  # Pasta de saída: output/PPO-{dims}D-{function.name}/
  # OBS:. Caso já exista, a saída é sobrescrita.
  training_utils.save_policy('PPO',
                             function,
                             dims,
                             agent.policy)


if __name__ == '__main__':
  multiprocessing.enable_interactive_mode()
  multiprocessing.handle_main(main)
