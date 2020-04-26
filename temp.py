# import tensorflow as tf
# import numpy as np
import os
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
# def convert_input(inputs, idx):
#     # # print(inputs[:, :idx])
#     # a = tf.math.reduce_sum(tf.math.log(inputs[:, :idx]), axis=1, keepdims=True)
#     # # print('a', a)
#     # b = tf.math.reduce_sum(tf.math.log(1 - inputs[:, idx+1:]), axis=1, keepdims=True)
#     # # print('a :', a, 'b:', b)
#     # c = a + b
#     # # print('c :', c)
#     # print(b)
#     return tf.math.reduce_sum(tf.math.log(inputs[:, :idx]), axis=1, keepdims=True) + tf.math.reduce_sum(tf.math.log(1 - inputs[:, idx + 1:]), axis=1, keepdims=True)
#
# #
# # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# bins = 3
# inputs = tf.constant([[-1,0,1]], dtype=tf.float32)
# inputs = tf.nn.sigmoid(inputs)
# print('Original input: ', inputs)
# # of size [batchsize, num-actions, bins]
# inputs = tf.expand_dims(inputs, axis=-1)
# print(inputs)
# inputs = tf.tile(inputs, [1, 1, 3])
# print("tiled: ", inputs)
# # # construct the mask
# left_diag = tf.constant(np.triu(np.ones((bins, bins)), 0), dtype=np.float32)
# left_diag = tf.expand_dims(left_diag, axis=0)
#
# right_diag = tf.constant(np.tril(np.ones((bins, bins)), -1), dtype=np.float32)
# right_diag = tf.expand_dims(right_diag, axis=0)
#
#
# a = tf.math.log(inputs) * left_diag
# b = tf.math.log(1 - inputs) * right_diag
# print('a :', a)
# print('b :', b)
# # c = a + b
# # print('c :', c)
# # c = tf.reduce_sum(c, axis=-1)
# # print('c :', c)
# pdparam = tf.reduce_sum(a + b, axis=-1)
# #pdparam = tf.nn.softmax(pdparam)
# print(pdparam)

# am_numpy = construct_mask(bins)
# am_tf = tf.constant(am_numpy, dtype=tf.float32)
#
# # construct pdparam
# pdparam = tf.reduce_sum(
#     tf.math.log(norm_softm_tiled + 1e-8) * am_tf + tf.math.log(1 - norm_softm_tiled + 1e-8) * (1 - am_tf), axis=-1)
# pdparam = tf.reshape(pdparam, [nbatch, -1])


# def convert_input(inputs, idx):
#     idx = tf.cast(idx, dtype=tf.int32)
#     return tf.math.reduce_sum(tf.math.log(inputs[:, :idx]), axis=1) + tf.math.reduce_sum(
#         tf.math.log(1 - inputs[:, idx + 1:]), axis=1)

def construct_mask(bins):
    a = np.zeros([bins,bins])
    for i in range(bins):
        for j in range(bins):
            if i+j <= bins-1:
                a[i,j] = 1.0
    return a
bins = 7
nbatch=1
inputs = tf.constant([[0.1,-0.1,0.1,-0.1,0,0.01,-0.2]], dtype=tf.float32)

inputs = tf.nn.sigmoid(inputs)  # of size [batchsize, num-actions*bins], initialized to be about uniform
        # output1 = tf.nn.softmax(inputs)
        #
inputs = tf.reshape(inputs, [1,-1,bins])
# inputs = tf.nn.sigmoid(inputs)
# inputs = tf.expand_dims(inputs, axis=1)
am_numpy = construct_mask(bins)

am_tf = tf.constant(am_numpy, dtype=tf.float32)

inputs = tf.tile(tf.expand_dims(inputs, axis=-1), [1, 1, 1, bins])
inputs = tf.reduce_sum(tf.math.log(inputs+1e-8) * am_tf + tf.math.log(1 - inputs + 1e-8) * (1 - am_tf),
                        axis=-1)
inputs = tf.reshape(inputs, [nbatch, -1])
dist = tfp.distributions.Categorical(logits=inputs)
list = dist.sample(1000)
list_plot = np.asarray(list)


fig_1 = plt.gcf()
plt.subplot(121)
# plt.bar(np.arange(len(inputs[0])),inputs[0])
plt.hist(list_plot)
plt.grid(True)
plt.subplot(122)
# #print(teamdata[0][i])
plt.bar(np.arange(len(inputs[0])),inputs[0])
# plt.subplot(133)
#print(teamdata[0][i])
# plt.bar(np.arange(len(output3[0])),output3[0])
plt.show()


# inputs = tf.nn.sigmoid(inputs)  # of size [batchsize, num-actions*bins], initialized to be about uniform
# output_dennis = tf.math.cumsum(tf.math.log(inputs+ 1e-8), axis=-1) + tf.math.cumsum(tf.math.log(1 - inputs+ 1e-8), axis=-1, reverse=True, exclusive=True)
# output_dennis=tf.reverse(output_dennis, [-1])

# output1 = tf.nn.softmax(inputs)
#
# inputs = tf.nn.sigmoid(inputs)
#
# am_numpy = construct_mask(inputs.shape[-1])
# am_tf = tf.constant(am_numpy, dtype=tf.float32)
# inputs = tf.tile(tf.expand_dims(inputs, axis=-1), [1, 1, inputs.shape[-1]])
# # construct pdparam
# print(1 - am_tf)
# pdparam = tf.reduce_sum(tf.math.log(inputs + 1e-8) * am_tf + tf.math.log(1 - inputs + 1e-8) * (1 - am_tf), axis=-1)
# print("pdparam : ", pdparam)
# print('Dennis :', output_dennis)
# output1 = tf.nn.softmax(output_dennis)
# output2 = tf.nn.softmax(pdparam)

# """"""
# idx = np.arange(len(inputs[0]), dtype=np.float32)
#
#
#
# # a = tf.math.reduce_sum(tf.math.log(inputs[:, :idx]), axis=1)
# # b = tf.math.reduce_sum(tf.math.log(1 - inputs[:, idx + 1:]), axis=1)
# # print('a :', a, 'b:', b)
#
# outputs = tf.map_fn(lambda idx: convert_input(inputs, idx), idx)
# outputs = tf.keras.backend.transpose(outputs)
# output3 = tf.nn.softmax(outputs)
# print(outputs)
#
#

# #
# # left_diag = tf.constant(np.tril(np.ones((inputs.shape[1], inputs.shape[1])), 0), dtype=np.float32)
# # right_diag = tf.constant(np.tril(np.ones((inputs.shape[1],inputs.shape[1])),-1).transpose((1,0)),dtype=np.float32)
# # left = tf.math.log(inputs+1e-8)
# right = tf.math.log(tf.constant(1, dtype=np.float32) - inputs+1e-8)
# left = tf.keras.backend.dot(left_diag, tf.keras.backend.transpose(left))
# right = tf.keras.backend.dot(right_diag, tf.keras.backend.transpose(right))
#
# output = tf.keras.backend.transpose(left+right)
# print(outpu


# print(inputsuts)
#
# inputs = tf.math.sigmoid(inputs)
# # print("Printing input clean :", inputs)
# idx = tf.range(0, 5, dtype=tf.int32)
#
# inputs = tf.map_fn(lambda idx: convert_input(inputs, idx), idx, dtype=tf.float32)
# print("final", inputs)
#
# idx = tf.cast(idx, dtype=tf.int32)
# # from config_ml import Config
# # import pickle
# # test = {'save_file_name': 'test',
# #         'n_ac': 5,
# #         'training_enabled': True,
# #         'n_neighbours': 3,
# #         'min_lat': 20,
# #         'max_lat': 50,
# #         'min_lon': 20,
# #         'max_lon': 50
# #         }
# #
# # settings = Config()
# # # print(settings.n_ac)
# # # settings.check()
# # settings.set_val(test)
# # settings.save_conf()
# # # print(settings.n_ac)
# # # settings.check()
# # settings2 = Config()
# # settings2 = settings.load_conf('test')
# #
# #
#
# # settings2 = Config()
# # settings2 = settings2.load_conf('test')
#
# # # from bluesky_env_ray import BlueSkyEnv
# # #
# # # class EnvConfigBuild:
# # #     def __init__(self, worker_index, vector_index):
# # #         self.worker_index = worker_index
# # #         self.vector_index = vector_index
# # #
# # #
# # # env_config = EnvConfigBuild(1, 1)
# # #
# # #
# # # env = BlueSkyEnv(env_config)
# # #
# # #
# # # env.reset()
# # #
# # # for i in range(100):
# # #     env.step(60)
# # import os
# # from gym import spaces
# # import numpy as np
# #
# # import ray
# # from ray.rllib.env.external_env import ExternalEnv
# # from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
# # from ray.rllib.utils.policy_server import PolicyServer
# # from ray.tune.logger import pretty_print
# # from ray.tune.registry import register_env
# # from ray.rllib.agents.ppo import PPOAgent
# #
# # SERVER_ADDRESS = "localhost"
# # SERVER_PORT = 27802
# # CHECKPOINT_FILE = "last_checkpoint.out"
# #
# # low_obs = np.array([40, 0, 0,0])
# # high_obs = np.array([60, 10, 360,1000])
# # observation_space_single = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
# # action_space_single = spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32)
# # from mlcontrolc_server import BlueSkyServer
# #
# # ray.init()
# # register_env("srv", lambda _: BlueSkyServer(action_space_single, observation_space_single))
# #
# #     # We use DQN since it supports off-policy actions, but you can choose and
# #     # configure any agent.
# #     # dqn = PPOAgent(
# #     #     env="srv",
# #     #     config={
# #     #         # Use a single process to avoid needing to set up a load balancer
# #     #         "num_workers": 0,
# #     #         # Configure the agent to run short iterations for debugging
# #     #         # "exploration_fraction": 0.01,
# #     #         # "learning_starts": 100,
# #     #         "timesteps_per_iteration": 500,
# #     #     })
# # trainer = PPOAgent(
# #     env="srv",
# #     config={
# #         "log_level": "INFO",
# #         'num_workers': 0,
# #         "vf_share_layers": True,
# #         # 'ignore_worker_failures': True,
# #         # 'num_cpus_per_worker':16,
# #         'num_envs_per_worker': 1,
# #         # 'env_config': {'nr_nodes': 12},
# #         # 'horizon': 500,
# #         'batch_mode': 'complete_episodes',
# #         'model': {
# #             'fcnet_hiddens': [256, 256],
# #             "use_lstm": False
# #         },
# #         'sample_batch_size': 200,
# #         'train_batch_size': 4000,
# #         'vf_clip_param': 50
# #
# #     })
# # # Attempt to restore from checkpoint if possible.
# # if os.path.exists(CHECKPOINT_FILE):
# #     checkpoint_path = open(CHECKPOINT_FILE).read()
# #     print("Restoring from checkpoint path", checkpoint_path)
# #     trainer.restore(checkpoint_path)
# #
# # # Serving and training loop
# # while True:
# #     print(pretty_print(trainer.train()))
# #     checkpoint_path = trainer.save()
# #     print("Last checkpoint", checkpoint_path)
# #     with open(CHECKPOINT_FILE, "w") as f:
# #         f.write(checkpoint_path)