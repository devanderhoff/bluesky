import os
from gym import spaces
import numpy as np

import ray
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.utils.policy_server import PolicyServer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOAgent

SERVER_ADDRESS = "localhost"
SERVER_PORT = 27802
CHECKPOINT_FILE = "test2.out"

low_obs = np.array([40, 0, 0,0])
high_obs = np.array([60, 10, 360,1000])
observation_space_single = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
action_space_single = spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32)

class BlueSkyServer(ExternalEnv):
    def __init__(self, action_space_single, observation_space_single):
        ExternalEnv.__init__(
            self, action_space_single,
            observation_space_single)

    def run(self):
        print("Starting policy server at {}:{}".format(SERVER_ADDRESS,
                                                       SERVER_PORT))
        server = PolicyServer(self, SERVER_ADDRESS, SERVER_PORT)
        server.serve_forever()

#multi agents obs: lat, long, hdg, dist_wpt, hdg_wpt, dist_plane1, hdg_plane1, dist_plane2, hdg_plane2 (nm/deg)
low_obs = np.array([40, 0, 0, 4, 0, 4, 0, 4, 0])
high_obs = np.array([60, 10, 360, 1000, 360, 1000, 360, 1000, 360])
observation_space_multi = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
action_space_multi = spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32)

class BlueSkyServerMultiAgent(ExternalMultiAgentEnv):
    def __init__(self, action_space_multi, observation_space_multi):
        ExternalEnv.__init__(
            self, action_space_multi,
            observation_space_multi)

    def run(self):
        print("Starting policy server at {}:{}".format(SERVER_ADDRESS,
                                                       SERVER_PORT))
        server = PolicyServer(self, SERVER_ADDRESS, SERVER_PORT)
        server.serve_forever()




if __name__ == "__main__":
    ray.init()
    register_env("srv", lambda _: BlueSkyServer(action_space_single, observation_space_single))

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
    trainer = PPOAgent(
        env="srv",
        config={
            "log_level": "INFO",
            'num_workers': 0,
            "vf_share_layers": False,
            # 'ignore_worker_failures': True,
            # 'num_cpus_per_worker':16,
            'num_envs_per_worker': 1,
            # 'env_config': {'nr_nodes': 12},
            # 'horizon': 500,
            'batch_mode': 'complete_episodes',
            'model': {
                'fcnet_hiddens': [256, 256],
                "use_lstm": False
            },
            'sample_batch_size': 5000,
            'train_batch_size': 80000,
            'vf_clip_param': 50

        })
    # Attempt to restore from checkpoint if possible.
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_path = open(CHECKPOINT_FILE).read()
        print("Restoring from checkpoint path", checkpoint_path)
        trainer.restore(checkpoint_path)

    # Serving and training loop
    while True:
        print(pretty_print(trainer.train()))
        checkpoint_path = trainer.save()
        print("Last checkpoint", checkpoint_path)
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(checkpoint_path)