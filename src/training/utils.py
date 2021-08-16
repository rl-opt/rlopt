"""Funções utilitárias para o treinamento dos agentes."""

import os

from tensorflow import io as tf_io

from tf_agents.policies import tf_policy
from tf_agents.policies import policy_saver

from src.functions import core
from src import config

ROOT_DIR = config.ROOT_DIR


def save_policy(algorithm_name: str,
                function: core.Function,
                dims: int,
                policy: tf_policy.TFPolicy):
  output_dir = os.path.join(ROOT_DIR,
                            'output',
                            f'{algorithm_name}-{str(dims)}D-{function.name}')
  tf_io.gfile.makedirs(output_dir)

  tf_policy_saver = policy_saver.PolicySaver(policy)
  tf_policy_saver.save(output_dir)
