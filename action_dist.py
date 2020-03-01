# import gym
# from gym.spaces import Discrete, Tuple
# import argparse
# import random
#
# import ray
# from ray import tune
# from ray.rllib.models import ModelCatalog
# from ray.rllib.models.tf.tf_action_dist import Categorical, TFActionDistribution
# from ray.rllib.models.tf.misc import normc_initializer
# from ray.rllib.models.tf.tf_modelv2 import TFModelV2
# from ray.rllib.utils.tuple_actions import TupleActions
# from ray.rllib.utils import try_import_tf
#
# tf = try_import_tf()
#
# class BetaDistribution(TFActionDistribution):
#
#     def __init__(self, inputs, model):
#         alpha, beta = tf.split(inputs, 2, axis=1)
#         self.alpha = alpha
#         self.beta = beta
#         super().__init__(inputs, model)
#
#
#     def _build_sample_op(self):
#         pass
#

