# from spinup import td3
import tensorflow as tf
import multiprocessing
import time
import gym
import gym_bluesky
# from spinup.utils.test_policy import load_policy, run_policy
import bluesky as bs
# from bluesky import tools
from bluesky.network.client import Client

from stable_baselines.common.policies import MlpLnLstmPolicy, MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2
# import stable_baselines

def main():

    # n_cpu = 8
    # env = SubprocVecEnv([make_env(i, n_cpu) for i in range(n_cpu)])
    #
    # policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[128,128, dict(vf=[128,128], pi=[128,128])])
    # model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='/home/dennis/tensorboard/PPO2_2e6', n_steps=500, learning_rate=0.003, vf_coef= 0.8, noptepochs=4, nminibatches=4, full_tensorboard_log=True, policy_kwargs=policy_kwargs,ent_coef=0.01)
    # model.learn(total_timesteps=2000000)
    # model.save("PPO2_1")

    model = PPO2.load("PPO2_1")
    env = gym.make('bluesky-v0', NodeID=0)
    for i_episode in range(20):
        obs = env.reset()
        for t in range(5000):
            action, _states = model.predict(obs)
            # print(action)
            obs, rewards, dones, info = env.step(action)
            if dones:
                print("Episode finished after {} timesteps".format(t+1))
                break






def make_env(node_id, n_cpu):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make('bluesky-v0', NodeID=node_id, n_cpu=n_cpu)
        return env

    return _init
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

