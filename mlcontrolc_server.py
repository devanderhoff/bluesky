import ray
import gym
import os
import numpy as np
from config_ml import Config
from action_dist import BetaDistributionAction, CategoricalOrdinal, CategoricalOrdinalTFP
from model import MyModelCentralized
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.env.policy_server_input import PolicyServerInput

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
# from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
    # , KLCoeffMixin, \
    # PPOLoss
# from ray.rllib.policy.tf_policy import LearningRateSchedule, \
#     EntropyCoeffSchedule, ACTION_LOGP

from ray.rllib.models import ModelCatalog
import time
# from Centralized import CentralizedValueMixin, centralized_critic_postprocessing, loss_with_central_critic, setup_mixins, central_vf_stats
from Centralized import centralized_critic_postprocessing, loss_with_central_critic, setup_mixins, central_vf_stats,\
    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, CentralizedValueMixin, CCPPO, CCTrainer
import Centralized

from ray import tune
# from ray.rllib.models.tf.tf_action_dist import SquashedGaussian
restore = False
##Config dict
config = {'save_file_name': 'config_file',
            'checkpoint_file_name': 'Centralized_high_LOS_test_8_new.out',
          'training_enabled': True,
          'multiagent': True,
          'server_port': 27800,
          'max_timesteps': 1500,
          'n_ac': 5, #20,
          'n_neighbours':4, #6,
          'min_lat': 20,
          'max_lat': 80,
          'min_lon': -60,
          'max_lon': 60,
          'min_lat_gen':49,#51
          'max_lat_gen':56,#54
          'min_lon_gen':0,#2
          'max_lon_gen':10,#8
          'min_dist': -2,
          'max_dist': 3000,
          'max_concurrent': 100,
          'lat_eham': 52.19,
          'lon_eham': 4.42,
          'wpt_reached': 5,
          'los': 8, #10
          'gamma':0.99
          }

settings = Config()
settings.set_val(config)

SERVER_ADDRESS = "localhost"
server_port = 27800
CHECKPOINT_FILE = "afsaga.out"

class BlueSkytest(ExternalEnv):
    def __init__(self, action_space_multi, observation_space_multi, env_config):
        ExternalEnv.__init__(self, action_space_multi, observation_space_multi, max_concurrent=100)

        self.env_config = env_config
        # self.server_port = env_config['server_port']
        # print(env_config.worker_index)
    def run(self):
        return


class BlueSkyServerMultiAgent(ExternalMultiAgentEnv):
    def __init__(self, action_space_multi, observation_space_multi, max_concurrent, env_config):
        ExternalMultiAgentEnv.__init__(
            self, action_space_multi,
            observation_space_multi, max_concurrent)
        self.env_config = env_config
        # self.server_port = env_config['server_port']
        # print(env_config.worker_index)

    def run(self):
        time.sleep(999999)
        # SERVER_PORT_new = self.server_port + self.env_config.worker_index
        # print("Starting policy server at {}:{}".format(SERVER_ADDRESS,
        #                                                SERVER_PORT_new))
        # server = PolicyServerInput(self, SERVER_ADDRESS, SERVER_PORT_new)
        # server.serve_forever()


if __name__ == "__main__":



    ModelCatalog.register_custom_model("Centralized", MyModelCentralized)
    # ModelCatalog.register_custom_action_dist("BetaDistributionAction", BetaDistributionAction)
    # ModelCatalog.register_custom_action_dist("CategoricalOrdinal", CategoricalOrdinal)
    ModelCatalog.register_custom_action_dist("CategoricalOrdinalTFP", CategoricalOrdinalTFP)

    # CCPPO = PPOTFPolicy.with_updates(
    #     name="CCPPO",
    #     postprocess_fn=centralized_critic_postprocessing,
    #     loss_fn=loss_with_central_critic,
    #     before_loss_init=setup_mixins,
    #     grad_stats_fn=central_vf_stats,
    #     mixins=[
    #         LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
    #         CentralizedValueMixin
    #     ])
    #
    # CCTrainer = PPOTrainer.with_updates(name="CCPPOTrainer", default_policy=CCPPO)

    if settings.multiagent:
        # Standard S0
        low_obs = np.array([settings.min_lat, settings.min_lon, 0, settings.min_dist, -180])  # -1, 0, -1, 0])
        high_obs = np.array(
            [settings.max_lat, settings.max_lon, 360, settings.max_dist, 180])  # , 360, 1000, 360, 1000, 360])

        fill_low_hdg = np.array([-180])
        fill_low_dist = np.array([settings.min_dist])
        fill_low = np.concatenate([fill_low_dist, fill_low_hdg])
        fill_low_obs = np.array([])

        for i in range(settings.n_neighbours):
            fill_low_obs = np.hstack([fill_low_obs, fill_low])

        fill_dist = np.array([settings.max_dist])
        fill_hdg = np.array([180])
        fill = np.concatenate([fill_dist, fill_hdg])
        high_obs_fill = np.array([])

        for i in range(settings.n_neighbours):
            high_obs_fill = np.hstack([high_obs_fill, fill])

        low_obs = np.concatenate([low_obs, fill_low_obs])
        high_obs = np.concatenate([high_obs, high_obs_fill])
        observation_space_multi = gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        # action_space_multi = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        action_space_multi = gym.spaces.Discrete(7)
    #
    # def env_creator(env_config):
    #     return BlueSkyServerMultiAgent(action_space_multi, observation_space_multi, settings.max_concurrent, env_config)
    # register_env("srv", lambda _: BlueSkyServer(action_space_single, observation_space_single))
    # register_env("srv", lambda env_config: BlueSkyServerMultiAgent(action_space_multi_discrete, observation_space_multi, max_concurrent, env_config))
    # ModelCatalog.register_custom_model("srv", BlueSkyServerMultiAgent)
    register_env("BlueSkySrv", lambda env_config: BlueSkyServerMultiAgent(action_space_multi, observation_space_multi, settings.max_concurrent, env_config))
    # register_env("srv", env_creator)
    # register_env("test", lambda env_config: ExternalEnv(action_space_multi, observation_space_multi, env_config))

    # connector_config = {
    #     # Use the connector server to generate experiences.
    #     "input": (
    #         lambda ioctx: PolicyServerInput( \
    #             ioctx, SERVER_ADDRESS, server_port)
    #     ),
    #     # Use a single worker process to run the server.
    #     "num_workers": 0,
    #     # Disable OPE, since the rollouts are coming from online clients.
    #     "input_evaluation": [],
    # }

    #
    ray.init()

    if True:
        trainer = CCTrainer(
            env='BlueSkySrv',
            config={
                "input": (
                    lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, server_port)),
                'model': {
                    "custom_model": "Centralized",
                    'custom_action_dist': 'CategoricalOrdinalTFP'},
                "input_evaluation": [],
                "log_level": "DEBUG",
                'num_workers': 0,
                'num_sgd_iter':5,
                'rollout_fragment_length' : 1000, #600
                'train_batch_size' : 60000, #12000
                'sgd_minibatch_size': 500,
                'batch_mode': 'complete_episodes',
                'num_gpus':1,
                'gamma': settings.gamma,
                'eager': False,
                # 'eager_tracing': True,
                'use_gae': True,
                'lr':1e-5,#normally e-6
                "explore": True,
                'lambda': settings.gamma,
            })
            # config={
            #     'env': 'BlueSkySrv',
            #     "input": (
            #         lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, server_port)),
            #     "input_evaluation": [],
            #     'lambda': 0.95,
            #     'kl_coeff': 0.5,
            #     'clip_rewards': True,
            #     'clip_param': 0.1,
            #     'vf_clip_param': 10.0,
            #     'entropy_coeff': 0.01,
            #     'train_batch_size': 50000,
            #     'sample_batch_size': 1000,
            #     'sgd_minibatch_size': 25000,
            #     'num_sgd_iter': 40,
            #     'num_workers': 0,
            #     # 'num_envs_per_worker': 5,
            #     'batch_mode': 'complete_episodes',
            #     'observation_filter': 'NoFilter',
            #     'vf_share_layers': 'true',
            #     'num_gpus': 1,
            #     'log_level': 'DEBUG'
            # }
            # )
        # Attempt to restore from checkpoint if possible.
        if os.path.exists(settings.checkpoint_file_name) and restore:
            checkpoint_path = open(settings.checkpoint_file_name).read()
            print("Restoring from checkpoint path", checkpoint_path)
            trainer.restore(checkpoint_path)

        # Serving and training loop
        while True:
            print(pretty_print(trainer.train()))
            checkpoint_path = trainer.save()
            print("Last checkpoint", checkpoint_path)
            with open(settings.checkpoint_file_name, "w") as f:
                f.write(checkpoint_path)

	

#
# # if not settings.multiagent:
# #     low_obs = np.array([20, -20, 0, 0, -180])  # -1, 0, -1, 0])
# #     high_obs = np.array([80, 20, 360, 1000, 180])  # , 360, 1000, 360, 1000, 360])
# #     fill_low = np.zeros(settings.n_neighbours * 2)
# #     print(fill_low)
# #
# #     fill_dist = np.array([1000])
# #     fill_hdg = np.array([180])
# #     fill = np.concatenate([fill_dist, fill_hdg])
# #     print(fill)
# #     high_obs_fill = np.array([])
# #
# #     for i in range(settings.n_neighbours):
# #         high_obs_fill = np.hstack([high_obs_fill, fill])
# #
# #     print(high_obs_fill)
# #     low_obs = np.concatenate([low_obs, fill_low])
# #     high_obs = np.concatenate([high_obs, high_obs_fill])
# #     observation_space_multi = gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
# #     # action_space_multi = gym.spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32)
# #     action_space_multi_discrete = gym.spaces.Discrete(5)
# #
# #     class BlueSkyServer(ExternalEnv):
# #         def __init__(self, action_space_single, observation_space_single):
# #             ExternalEnv.__init__(
# #                 self, action_space_single,
# #                 observation_space_single)
# #
# #         def run(self):
# #             print("Starting policy server at {}:{}".format(SERVER_ADDRESS,
# #                                                            server_port))
# #             server = PolicyServer(self, SERVER_ADDRESS, server_port)
# #             server.serve_forever()
# #
# # #multi agents obs: lat, long, hdg, dist_wpt, hdg_wpt, dist_plane1, hdg_plane1, dist_plane2, hdg_plane2 (nm/deg)
# #
#PPOTrainer(
#         "PPO",
# config={
#     'env': 'BlueSkySrv',
#     "input": (
#         lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, server_port)),
#     'lambda': 0.95,
#     'kl_coeff': 0.5,
#     'clip_rewards': True,
#     'clip_param': 0.1,
#     'vf_clip_param': 10.0,
#     'entropy_coeff': 0.01,
#     'train_batch_size': 5000,
#     'sample_batch_size': 100,
#     'sgd_minibatch_size': 500,
#     'num_sgd_iter': 10,
#     'num_workers': 0,
#     'num_envs_per_worker': 5,
#     'batch_mode': 'truncate_episodes',
#     'observation_filter': 'NoFilter',
#     'vf_share_layers': 'true',
#     'num_gpus': 1,
#     'log_level': 'DEBUG'
# }
# #     )

# PPOTrainer(
#         "PPO",
#         config={
#             'env': 'BlueSkySrv',
#             "input": (
#                 lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, server_port)),
#             'model': {
#                 "custom_model": "Centralized",
#                 'custom_action_dist': 'CategoricalOrdinalTFP'},
#             # 'model': {'custom_action_dist': 'SquashedGaussian',
#             #           },
#             "input_evaluation": [],
#             "log_level": "DEBUG",
#             'num_workers': 0,
#             # "vf_share_layers": True,
#             # "vf_loss_coeff" : 1e-6,
#             # 'ignore_worker_failures': True,
#             # 'num_cpus_per_worker':3,
#             # 'num_cpus_for_driver':1,
#             # 'num_envs_per_worker': 1,
#             # 'env_config': {'server_port' : 27800},
#             'rollout_fragment_length': 1000,  # 600
#             'train_batch_size': 200000,  # 12000
#             'num_sgd_iter':1,
#             # 'clip_actions': True,
#             # 'env_config': {'nr_nodes': 12},
#             # 'horizon': 500,
#             'batch_mode': 'complete_episodes',
#             # 'observation_filter': 'MeanStdFilter',
#             'num_gpus': 1,
#             # 'num_gpus_per_worker':0.2,
#             'gamma': settings.gamma,
#             'eager': False,
#             # 'eager_tracing':True,
#             # 'model': {
#             #      'fcnet_hiddens': [256, 256],
#             # "use_lstm": True
#             # },
#             # 'sample_batch_size': 5000,
#             # 'train_batch_size': 80000,
#             # 'vf_clip_param': 50
#             'use_gae': True,
#             # "kl_target": 0.01,
#             # 'kl_coeff':0.4,
#             # 'entropy_coeff': 0.1,
#             # "shuffle_sequences": False,
#             'lr': 1e-5,  # normally e-6
#             "explore": True,
#             # "exploration_config": {
#             #     "type": "SoftQ",
#             #     # Parameters for the Exploration class' constructor:
#             #     "temperature": 1.0},
#             'lambda': settings.gamma,
#             # 'no_done_at_end': True
#         },
#     )
