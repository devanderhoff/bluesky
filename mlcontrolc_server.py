import gym
import numpy as np
import sys, os
from config_ml import Config

from model import MyModelCentralized
import ray
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.utils.policy_server import PolicyServer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import SquashedGaussian

##Config dict
config = {'save_file_name': 'config_file1',
'checkpoint_file_name': 'testrun5.out',
          'training_enabled': True,
          'multiagent': True,
          'server_port': 27800,
        'max_timesteps': 500,
          'n_ac': 3,
          'n_neighbours': 3,
          'min_lat': 20,
          'max_lat': 80,
          'min_lon': -20,
          'max_lon': 20,
          'min_lat_gen':51,
          'max_lat_gen':54,
          'min_lon_gen':2,
          'max_lon_gen':8,
          'min_dist': 0,
          'max_dist': 3000,
          'max_concurrent': 100,
          'lat_eham': 52.19,
          'lon_eham': 4.42,
          'wpt_reached': 5,
          'los': 2.5,
          'gamma':0.99


          }

settings = Config()
settings.set_val(config)

SERVER_ADDRESS = "localhost"
server_port = 27800
CHECKPOINT_FILE = "afsaga.out"



if not settings.multiagent:
    low_obs = np.array([20, -20, 0, 0, -180])  # -1, 0, -1, 0])
    high_obs = np.array([80, 20, 360, 1000, 180])  # , 360, 1000, 360, 1000, 360])
    fill_low = np.zeros(settings.n_neighbours * 2)
    print(fill_low)

    fill_dist = np.array([1000])
    fill_hdg = np.array([180])
    fill = np.concatenate([fill_dist, fill_hdg])
    print(fill)
    high_obs_fill = np.array([])

    for i in range(settings.n_neighbours):
        high_obs_fill = np.hstack([high_obs_fill, fill])

    print(high_obs_fill)
    low_obs = np.concatenate([low_obs, fill_low])
    high_obs = np.concatenate([high_obs, high_obs_fill])
    observation_space_multi = gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
    # action_space_multi = gym.spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32)
    action_space_multi_discrete = gym.spaces.Discrete(5)

    class BlueSkyServer(ExternalEnv):
        def __init__(self, action_space_single, observation_space_single):
            ExternalEnv.__init__(
                self, action_space_single,
                observation_space_single)

        def run(self):
            print("Starting policy server at {}:{}".format(SERVER_ADDRESS,
                                                           server_port))
            server = PolicyServer(self, SERVER_ADDRESS, server_port)
            server.serve_forever()

#multi agents obs: lat, long, hdg, dist_wpt, hdg_wpt, dist_plane1, hdg_plane1, dist_plane2, hdg_plane2 (nm/deg)


class BlueSkyServerMultiAgent(ExternalMultiAgentEnv):
    def __init__(self, action_space_multi, observation_space_multi, max_concurrent, env_config):
        ExternalEnv.__init__(
            self, action_space_multi,
            observation_space_multi, max_concurrent)
        self.env_config = env_config
        self.server_port = env_config['server_port']
        print(env_config.worker_index)

    def run(self):
        SERVER_PORT_new = self.server_port + self.env_config.worker_index
        print("Starting policy server at {}:{}".format(SERVER_ADDRESS,
                                                       SERVER_PORT_new))
        server = PolicyServer(self, SERVER_ADDRESS, SERVER_PORT_new)
        server.serve_forever()

if __name__ == "__main__":

    ray.init()

    ModelCatalog.register_custom_model("Centralized", MyModelCentralized)
    # ModelCatalog.register_custom_action_dist("SquashedGaussian", SquashedGaussian)


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
        action_space_multi = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # action_space_multi_discrete = gym.spaces.Discrete(5)

    # register_env("srv", lambda _: BlueSkyServer(action_space_single, observation_space_single))
    # register_env("srv", lambda env_config: BlueSkyServerMultiAgent(action_space_multi_discrete, observation_space_multi, max_concurrent, env_config))
    # ModelCatalog.register_custom_model("srv", BlueSkyServerMultiAgent)
    register_env("srv", lambda env_config: BlueSkyServerMultiAgent(action_space_multi, observation_space_multi, settings.max_concurrent, env_config))
    # We use DQN since it supports off-policy actions, but you can choose and
    # configure any agent.
    # dqn = PPOAgent(
    #     env="srv",
    #     config={
    #         # Use a single process to avoid needing to set up a load balancer
    #         "num_workers": 0,
    #         # Configure the agent to run short iterations for debugging
    #         # "exploration_fraction": 0.01,
    #         # "learning_starts": 100,
    #         "timesteps_per_iteration": 500,
    #     })

    if True:
        trainer = PPOTrainer(
            env="srv",
            config={
                'model': {"custom_model": "Centralized",
                          # 'custom_action_dist': 'SquashedGaussian'
                          },
                # 'model': {'custom_action_dist': 'SquashedGaussian',
                #           },
                # "log_level": "INFO",
                'num_workers': 0,
                # "vf_share_layers": True,
                # "vf_loss_coeff" : 1e-6,
                # 'ignore_worker_failures': True,
                # 'num_cpus_per_worker':16,
                'num_envs_per_worker': 1,
                'env_config': {'server_port' : 27800},
                'sample_batch_size' : 100,
                'train_batch_size' : 12000,
                # 'clip_actions': True,
                # 'env_config': {'nr_nodes': 12},
                # 'horizon': 500,
                'batch_mode': 'complete_episodes',
                # 'observation_filter': 'MeanStdFilter',
                'num_gpus':1,
                'gamma': settings.gamma,
                # 'model': {
                #      'fcnet_hiddens': [256, 256],
                     # "use_lstm": True
                 # },
                # 'sample_batch_size': 5000,
                # 'train_batch_size': 80000,
                # 'vf_clip_param': 50
                 'use_gae': True


            })
        # Attempt to restore from checkpoint if possible.
        if os.path.exists(settings.checkpoint_file_name):
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