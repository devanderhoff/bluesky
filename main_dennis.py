# from spinup import td3
# import tensorflow as tf
# import multiprocessing
# import time
import gym
# import gym_bluesky
# from spinup.utils.test_policy import load_policy, run_policy
import bluesky as bs
# from bluesky import tools
# from bluesky.network.client import Client

# from stable_baselines.common.policies import MlpLnLstmPolicy, MlpPolicy
# from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
# from stable_baselines import PPO2
# import stable_baselines
from ray.rllib.env import MultiAgentEnv

import numpy as np
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ddpg as ddpg


import ray
from ray.tune.registry import register_env
from ray import tune
#
# from ray.rllib.env.env_context import EnvContext

from bluesky_env_ray import BlueSkyEnv
# from EnvironmentExample import EnvironmentExample

class MultiEnv(MultiAgentEnv):
    def __init__(self, env_config):
        # pick actual env based on worker and env indexes
        self.env = BlueSkyEnv(env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)




def main():
    # bs.init()
    # env_config = {}
    # test = BlueSkyEnv(env_config)
    # test.reset()
    # # action = dict(SUP0=5)
    # for i in range(500):
    #     kaas = test.step(5)
    #     print('iteration loop nr: ' + str(i))
    #     print(kaas)
    # def make_env(i, n_cpu):
    #     def _init():
    #         env = gym.make('bluesky-v0', NodeID=i, n_cpu=n_cpu)
    #         return env
    #     return _init()

    #
    # # n_cpu = 8
    # # env = SubprocVecEnv([make_env(i, n_cpu) for i in range(n_cpu)])
    # #
    # # policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[128,128, dict(vf=[128,128], pi=[128,128])])
    # # model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='/home/dennis/tensorboard/PPO2_2e6', n_steps=500, learning_rate=0.003, vf_coef= 0.8, noptepochs=4, nminibatches=4, full_tensorboard_log=True, policy_kwargs=policy_kwargs,ent_coef=0.01)
    # # model.learn(total_timesteps=2000000)
    # # model.save("PPO2_1_222")
    # #
    # model = PPO2.load("PPO2_1")
    # env = gym.make('bluesky-v0', NodeID=0)
    # for i_episode in range(20):
    #     obs = env.reset()
    #     while True:
    #         action, _states = model.predict(obs)
    #         obs, rewards, dones, info = env.step(action)
    #         # env.render(/)
    #
    #     # obs, rewards, dones, info =/
    # #         if dones:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break
    # #

    #
    ################### NEWWWWWWWWWWW TEST ##################################
    # gym.envs.register(
    # id='bluesky-v0',
    # entry_point='gym_bluesky.envs:BlueSkyEnv',
    # kwargs={'NodeID': 0,
    #         'n_cpu': 1,
    #         'scenfile': None})




    # env_config = EnvConfig()
    # env_config = 'kaas'
    #
    #         'horizon':500,
    #         'batch_mode':'complete_episodes',
    # test = BlueSkyEnv(env_config)
    ray.init()
    # # # env_creator = lambda config:make_env(config,0,0)

    register_env("Bluesky", lambda config: MultiEnv(config))
    # #
    print('hallo2')
    # # # low_obs = np.array([-1,-1,-1,0,0,0])
    # # # high_obs = np.array([1,1,1,1,1,1])
    trainer = ppo.PPOAgent(env="Bluesky", config={
            "log_level":"DEBUG",
            'num_workers':4,
            "vf_share_layers":True,
            #'num_cpus_per_worker':16,
            'num_envs_per_worker':1,
            'env_config':{'nr_nodes':12},
            'horizon':2000,
            'batch_mode':'complete_episodes',
            'model':{
                'fcnet_hiddens':[256,256],
                "use_lstm":False
            },
            'sample_batch_size':200,
            'train_batch_size':4000,
            'vf_clip_param':50

        })
    for i in range(151):
        trainer.train()
        if i % 10 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint, i)
    print('hallo3')
    # trainer = ddpg.DDPGAgent(env="Bluesky", config={
    #         "log_level":"INFO",
    #         'num_workers':4,
    #         # "vf_share_layers":True,
    #         'num_cpus_per_worker':1,
    #         'num_envs_per_worker':4,
    #         'env_config':{'nr_nodes':12},
    #         'horizon':500,
    #         # 'batch_mode':'complete_episodes',
    #         # 'model':{
    #         #     'fcnet_hiddens':[256,256],
    #         #     "use_lstm":False
    #         # },
    #         # 'sample_batch_size':200,
    #         # 'train_batch_size':4000,
    #         # 'vf_clip_param':50
    #
    #     })
    # #
    # #
    # #
    # # trainer.restore('/home/dennis/ray_results/PPO_Bluesky_2019-04-11_22-31-19ug0rl6c9/checkpoint_243/checkpoint-243')
    #
    # for i in range(151):
    #     trainer.train()
    #     if i % 10 == 0:
    #         checkpoint = trainer.save()
    #         print("checkpoint saved at", checkpoint, i)
    # # obs_space = gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
    # # Action space is normalized heading, shape (1,)
    # # act_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    #
    #
    #
    # #
    # # tune.run(
    # #     "PPO",
    # #     name='Test3',
    # #     local_dir='~/ray_results/superduper2',
    # #     checkpoint_freq=5,
    # #     checkpoint_at_end=True,
    # #     verbose=2,
    # #     resume='prompt',
    # #     stop={"training_iteration": 10},
    # #     restore='/home/dennis/ray_results/superduper2/Test3/PPO_Bluesky_0_2019-04-11_19-40-37e5tat2oe',
    # #     config={
    # #         "env":"Bluesky",
    # #         "log_level":"DEBUG",
    # #         'num_workers':4,
    # #         "vf_share_layers":True,
    # #         'num_cpus_per_worker':1,
    # #         'num_envs_per_worker':4,
    # #         'env_config':{'nr_nodes':12},
    #         'model':{
    #             'fcnet_hiddens':[256,256],
    #             "use_lstm":True
    #         },
    #         'sample_batch_size':200,
    #         'train_batch_size':4000,
    #         'vf_clip_param':50
    #
    #     },)

    # trainer = ppo.PPOAgent(env="Bluesky")
    #     ,
    #                        config={
    #         "multiagent": {
    #             "policy_graphs": {
    #                 # the first tuple value is None -> uses default policy graph
    #                 "SUP0": (None, obs_space, act_space, {"gamma": 0.99}),
    #                 "SUP1": (None, obs_space, act_space, {"gamma": 0.99}),
    #                 "SUP2": (None, obs_space, act_space, {"gamma": 0.99}),
    #             },
    #             'policy_mapping_fn':
    #             lambda agent_id:
    #                 "SUP"
    #
    #
    #         },
    #                        })

    # while True:
    #     print(trainer.train())

    # tune.run(
    #     "PPO",
    #     stop={"training_iteration": 200},
    #     config={
    #         "env": bluesky_env.BlueSkyEnv,
    #         "log_level": "DEBUG",
    #         "num_sgd_iter": 10,
    #         "multiagent": {
    #             "policy_graphs": {
    #                 # the first tuple value is None -> uses default policy graph
    #                 "SUP0": (None, obs_space, act_space, {"gamma": 0.99}),
    #                 "SUP1": (None, obs_space, act_space, {"gamma": 0.99}),
    #                 "SUP2": (None, obs_space, act_space, {"gamma": 0.99})}}})


def env_creator(env_config):
    return BlueSkyEnv(env_config)


# #
# # #
# model = PPO2.load("testsgjklasgjkl30")
# model.set_env(env=env)
# model.learn(total_timesteps=200000, tb_log_name='/home/dennis/tensorboard/test8')
# model.save("test21")
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

# #
# for i_episode in range(20):
#     obs = env.reset()
#     for t in range(500):
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         # print(action)
#         env.render()
#         if dones:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
# # #
#

#
# # # #
#     env = gym.make('bluesky-v0', NodeID=0)
#     env2 = gym.make('bluesky-v0', NodeID=1)
#     env3 = gym.make('bluesky-v0', NodeID=2)
#     env4 = gym.make('bluesky-v0', NodeID=3)
# # # # #
#     for i_episode in range(10):
#         observation = env.reset()
#         observation2 = env2.reset()
#         observation3 = env3.reset()
#         observation4 = env4.reset()
#         for t in range(1000):
#             # env.render()
#
#             action = env.action_space.sample()
#             # print(action)
#             action2 = env2.action_space.sample()
#             action3 = env2.action_space.sample()
#             action4 = env2.action_space.sample()
#             observation, reward, done, info = env.step(action)
#             observation2, reward2, done2, info2 = env2.step(action2)
#             observation3, reward3, done3, info3 = env3.step(action3)
#             observation4, reward4, done4, info4 = env4.step(action4)
#
# # # # env.close()
# # # # #
# # # # # #
# # # #
#
# #
# env_fn = lambda : gym.make('bluesky-v0')
#
# ac_kwargs = dict(hidden_sizes=[128,128], activation=tf.nn.tanh)
# logger_kwargs = dict(output_dir='/home/dennis/spinupoutput2/run_25', exp_name='run_25')
#
# td3(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=20, logger_kwargs=logger_kwargs, max_ep_len=1000, start_steps=0)
#
#
# # Run 22 circle
# #
# _, get_action = load_policy('/home/dennis/spinupoutput2/run_24')
#
# env = gym.make('bluesky-v0')
# run_policy(env, get_action)



# up = tools.geo.kwikdist(-1,1,1,1)
# right = tools.geo.kwikdist(1,1,1,-1)
# bottom = tools.geo.kwikdist(-1,-1,1,-1)
# left = tools.geo.kwikdist(-1,1,-1,-1)
#
# print(up,right,bottom,left)
if __name__ == '__main__':
     main()

