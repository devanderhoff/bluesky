from config_ml import Config
import pickle
test = {'save_file_name': 'test',
        'n_ac': 5,
        'training_enabled': True,
        'n_neighbours': 3,
        'min_lat': 20,
        'max_lat': 50,
        'min_lon': 20,
        'max_lon': 50
        }

settings = Config()
# print(settings.n_ac)
# settings.check()
settings.set_val(test)
settings.save_conf()
# print(settings.n_ac)
# settings.check()
settings2 = Config()
settings2 = settings.load_conf('test')



# settings2 = Config()
# settings2 = settings2.load_conf('test')

# # from bluesky_env_ray import BlueSkyEnv
# #
# # class EnvConfigBuild:
# #     def __init__(self, worker_index, vector_index):
# #         self.worker_index = worker_index
# #         self.vector_index = vector_index
# #
# #
# # env_config = EnvConfigBuild(1, 1)
# #
# #
# # env = BlueSkyEnv(env_config)
# #
# #
# # env.reset()
# #
# # for i in range(100):
# #     env.step(60)
# import os
# from gym import spaces
# import numpy as np
#
# import ray
# from ray.rllib.env.external_env import ExternalEnv
# from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
# from ray.rllib.utils.policy_server import PolicyServer
# from ray.tune.logger import pretty_print
# from ray.tune.registry import register_env
# from ray.rllib.agents.ppo import PPOAgent
#
# SERVER_ADDRESS = "localhost"
# SERVER_PORT = 27802
# CHECKPOINT_FILE = "last_checkpoint.out"
#
# low_obs = np.array([40, 0, 0,0])
# high_obs = np.array([60, 10, 360,1000])
# observation_space_single = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
# action_space_single = spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32)
# from mlcontrolc_server import BlueSkyServer
#
# ray.init()
# register_env("srv", lambda _: BlueSkyServer(action_space_single, observation_space_single))
#
#     # We use DQN since it supports off-policy actions, but you can choose and
#     # configure any agent.
#     # dqn = PPOAgent(
#     #     env="srv",
#     #     config={
#     #         # Use a single process to avoid needing to set up a load balancer
#     #         "num_workers": 0,
#     #         # Configure the agent to run short iterations for debugging
#     #         # "exploration_fraction": 0.01,
#     #         # "learning_starts": 100,
#     #         "timesteps_per_iteration": 500,
#     #     })
# trainer = PPOAgent(
#     env="srv",
#     config={
#         "log_level": "INFO",
#         'num_workers': 0,
#         "vf_share_layers": True,
#         # 'ignore_worker_failures': True,
#         # 'num_cpus_per_worker':16,
#         'num_envs_per_worker': 1,
#         # 'env_config': {'nr_nodes': 12},
#         # 'horizon': 500,
#         'batch_mode': 'complete_episodes',
#         'model': {
#             'fcnet_hiddens': [256, 256],
#             "use_lstm": False
#         },
#         'sample_batch_size': 200,
#         'train_batch_size': 4000,
#         'vf_clip_param': 50
#
#     })
# # Attempt to restore from checkpoint if possible.
# if os.path.exists(CHECKPOINT_FILE):
#     checkpoint_path = open(CHECKPOINT_FILE).read()
#     print("Restoring from checkpoint path", checkpoint_path)
#     trainer.restore(checkpoint_path)
#
# # Serving and training loop
# while True:
#     print(pretty_print(trainer.train()))
#     checkpoint_path = trainer.save()
#     print("Last checkpoint", checkpoint_path)
#     with open(CHECKPOINT_FILE, "w") as f:
#         f.write(checkpoint_path)