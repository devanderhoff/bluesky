import os
from gym import spaces
import numpy as np

import ray
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.utils.policy_server import PolicyServer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOAgent

SERVER_ADDRESS = "localhost"
SERVER_PORT = 27802
CHECKPOINT_FILE = "last_checkpoint.out"

low_obs = np.array([40, 0, 0])
high_obs = np.array([60, 10, 360])
observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
action_space = spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32)

class BlueSkyServer(ExternalEnv):
    def __init__(self, observation_space, action_space):
        ExternalEnv.__init__(
            self, observation_space,
            action_space)

    def run(self):
        print("Starting policy server at {}:{}".format(SERVER_ADDRESS,
                                                       SERVER_PORT))
        server = PolicyServer(self, SERVER_ADDRESS, SERVER_PORT)
        server.serve_forever()




if __name__ == "__main__":
    ray.init()
    register_env("srv", lambda _: BlueSkyServer(observation_space, action_space))

    # We use DQN since it supports off-policy actions, but you can choose and
    # configure any agent.
    dqn = PPOAgent(
        env="srv",
        config={
            # Use a single process to avoid needing to set up a load balancer
            "num_workers": 0,
            # Configure the agent to run short iterations for debugging
            # "exploration_fraction": 0.01,
            # "learning_starts": 100,
            "timesteps_per_iteration": 500,
        })

    # Attempt to restore from checkpoint if possible.
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_path = open(CHECKPOINT_FILE).read()
        print("Restoring from checkpoint path", checkpoint_path)
        dqn.restore(checkpoint_path)

    # Serving and training loop
    while True:
        print(pretty_print(dqn.train()))
        checkpoint_path = dqn.save()
        print("Last checkpoint", checkpoint_path)
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(checkpoint_path)