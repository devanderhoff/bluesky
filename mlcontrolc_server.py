from config_ml import Config
"""
Disable KL-coeff, only use clipping x
Swap to determinisic sampling around 650M steps (misschien) 
Maximize batch size and mini batch size
Destination airport should be equally divided (np shuffle) x
Make delete dict dynnamic
Disable NavDB
RMSprop instead of ADAM x
TODO destination heading fucked

"""
config = {'save_file_name': 'config_file',
            'checkpoint_file_name': 'tastatsd.out',
          'training_enabled': True,
          'multiagent': True,
          'server_port': 27800,
          'max_timesteps': 1500, #1500
          'n_ac': 25, #20,
          'n_neighbours':5, #6,
          'min_lat': 20,
          'max_lat': 80,
          'min_lon': -60,
          'max_lon': 60,
          'min_lat_gen':51,#51 , 49
          'max_lat_gen':53,#54 , 56
          'min_lon_gen':2,#2 , 0
          'max_lon_gen':8,#8 , 10
          'min_dist': -2,
          'max_dist': 3000,
          'max_concurrent': 100,
          'lat_eham': 52.19,
          'lon_eham': 4.42,
          'wpt_reached': 3,
          'los': 5, #10
          'gamma':0.99,
          'spawn_separation':15,
          'destination_distribution':'uniform',
          'destination_hdg': True,
          'multi_destination': True,
          }

settings = Config()
settings.set_val(config)

import ray
import gym
import os
import numpy as np
import random

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
from ray.tune import run, sample_from
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.models import ModelCatalog
import time
# from Centralized import CentralizedValueMixin, centralized_critic_postprocessing, loss_with_central_critic, setup_mixins, central_vf_stats
from Centralized import centralized_critic_postprocessing, loss_with_central_critic, setup_mixins, central_vf_stats,\
    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, CentralizedValueMixin, CCPPO, CCTrainer
import Centralized
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray import tune
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from typing import Dict

restore = True
##Config dict

SERVER_ADDRESS = "localhost"
server_port = 27800
CHECKPOINT_FILE = "afsaga.out"

# Create "Env" for RLlib. They are mostly dummy env's, for compatibility reasons.


class BlueSkyServerMultiAgent(ExternalMultiAgentEnv):
    def __init__(self, action_space_multi, observation_space_multi, max_concurrent, env_config):
        ExternalMultiAgentEnv.__init__(
            self, action_space_multi,
            observation_space_multi, max_concurrent)
        self.env_config = env_config

    def run(self):
        # 116 days of training before this should raise an error.
        time.sleep(9999999)

        # Legacy since RLlib 8.5
        # SERVER_PORT_new = self.server_port + self.env_config.worker_index
        # print("Starting policy server at {}:{}".format(SERVER_ADDRESS,
        #                                                SERVER_PORT_new))
        # server = PolicyServerInput(self, SERVER_ADDRESS, SERVER_PORT_new)
        # server.serve_forever()

class MyCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        episode.custom_metrics['nr_ac_crashes'] = sum(value == -1 for value in episode.agent_rewards.values())
        episode.custom_metrics['nr_ac_landed_with_hdg'] = sum(value == 2 for value in episode.agent_rewards.values())
        episode.custom_metrics['nr_ac_landed_without_hdg']= sum(value == 1 for value in episode.agent_rewards.values())

if __name__ == "__main__":
    # First create action space and observation space bounds. This is required by OpenAI gym/RLlib.
    # TODO remove hardcoded action space and observation space.
    if settings.multiagent:
        # Standard S0
        low_obs = np.array([0,0, settings.min_lat, settings.min_lon, 0, settings.min_dist, -180])  # -1, 0, -1, 0])
        high_obs = np.array(
            [3,3, settings.max_lat, settings.max_lon, 360, settings.max_dist, 180])  # , 360, 1000, 360, 1000, 360])

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
        action_space_multi = gym.spaces.Discrete(7)

    # Register env's and custom stuff to RLlib for trainer to be able to use them.
    register_env("BlueSkySrv", lambda env_config: BlueSkyServerMultiAgent(action_space_multi, observation_space_multi,
                                                                          settings.max_concurrent, env_config))
    ModelCatalog.register_custom_model("Centralized", MyModelCentralized)
    # ModelCatalog.register_custom_action_dist("BetaDistributionAction", BetaDistributionAction)
    # ModelCatalog.register_custom_action_dist("CategoricalOrdinal", CategoricalOrdinal)
    ModelCatalog.register_custom_action_dist("CategoricalOrdinalTFP", CategoricalOrdinalTFP)

    # Init ray.
    ray.init()

    # def explore(config):
    #     # ensure we collect enough timesteps to do sgd
    #     if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
    #         config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    #     # ensure we run at least one sgd iter
    #     if config["num_sgd_iter"] < 1:
    #         config["num_sgd_iter"] = 1
    #     return config
    #
    # pbt = PopulationBasedTraining(
    #     time_attr="time_total_s",
    #     metric="episode_reward_mean",
    #     mode="max",
    #     perturbation_interval=120,
    #     resample_probability=0.25,
    #     # Specifies the mutations of these hyperparams
    #     hyperparam_mutations={
    #         "lambda": lambda: random.uniform(0.9, 1.0),
    #         "clip_param": lambda: random.uniform(0.01, 0.5),
    #         "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    #         "num_sgd_iter": lambda: random.randint(1, 15),
    #         "sgd_minibatch_size": lambda: random.randint(10000, 50000),
    #         "train_batch_size": lambda: random.randint(30000, 200000),
    #     },
    #     custom_explore_fn=explore
    #     )
    if True:
        run(
            CCTrainer,
            name="Fullscale_test_thursday_entr001_kl00001_100kbatch",
            # stop={"timesteps_total":100000000},
            # scheduler=pbt,
            checkpoint_freq=10,
            num_samples=1,
            # restore='~/ray_results/1-destination-lr1e-4-entr1e-3_try2/CCPPOTrainer_BlueSkySrv_0_2020-05-27_08-45-06wsu0r_f6/checkpoint_8110/checkpoint-8110',
            # with_server=True,
            # server_port=27800,
            # resources_per_trial={'gpu':1, 'cpu':1},
            config={
                "callbacks": MyCallbacks,
                "input": (
                    lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, server_port)),
                "input_evaluation": [],
                "log_level": "DEBUG",
                "env": "BlueSkySrv",
                "kl_coeff": 0,
                # "num_workers": 1,
                "model": {
                    "custom_model": "Centralized",
                    'custom_action_dist': 'CategoricalOrdinalTFP'
                },
                # These params are tuned from a fixed starting value.
                'entropy_coeff':0.01,#0.001
                "lambda": 0.95,
                "clip_param": 0.3, #0.2
                "lr": 0.0001,#0.0001
                'num_workers': 0,
                'num_sgd_iter': 10,
                'rollout_fragment_length': 1000,  # 600
                'train_batch_size': 100000,  # 80000
                'sgd_minibatch_size': 100000,
                'batch_mode': 'complete_episodes',
                'num_gpus': 1,
                'gamma': settings.gamma,
                'eager': False,
                # 'eager_tracing': True,
                'use_gae': True,
                # #ADAM schedule
                # 'lr_schedule':[
                #     [0, 0.0005],
                #     [800000000, 0.00001],
                # ],
                # RMSprop
                # 'lr_schedule': [
                #     [0, 0.0005],
                #     [250000000, 0.00001],
                # ],
                # 'kl_target': 0.03,
                "explore": True,
                'shuffle_sequences': True
                # These params start off randomly drawn from a set.
                # "num_sgd_iter": sample_from(
                #     lambda spec: random.choice([5, 10, 15])),
                # "sgd_minibatch_size": sample_from(
                #     lambda spec: random.choice([10000, 25000, 50000])),
                # "train_batch_size": sample_from(
                #     lambda spec: random.choice([50000, 100000, 200000]))
            })
    #Start training loop.
    if False:
        trainer = CCTrainer(
            env='BlueSkySrv',
            config={
                "input": (
                    lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, server_port)),
                'model': {
                    "custom_model": "Centralized",
                    'custom_action_dist': 'CategoricalOrdinalTFP'},
                "input_evaluation": [],
                "log_level": "INFO",
                'num_workers': 0,
                'num_sgd_iter':10,
                'rollout_fragment_length' : 1000, #600
                'train_batch_size' : 200000, #80000
                'sgd_minibatch_size': 100000,
                'batch_mode': 'complete_episodes',
                'num_gpus':1,
                'gamma': settings.gamma,
                'eager': False,
                # 'eager_tracing': True,
                'use_gae': True,
                'lr':0.0001,#normally e-6
                'entropy_coeff': 0.001,
                # #ADAM schedule
                # 'lr_schedule':[
                #     [0, 0.0005],
                #     [800000000, 0.00001],
                # ],
                #RMSprop
                # 'lr_schedule': [
                #     [0, 0.0005],
                #     [250000000, 0.00001],
                # ],
                'clip_param': 0.2, #was .3
                'kl_coeff':0,
                'kl_target': 0.03,
                # 'entropy_coeff': 0.01, #uit later
                # 'opt_type':'Adam',
                "explore": True,
                'lambda': 0.95,
                'shuffle_sequences': True
            })

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


