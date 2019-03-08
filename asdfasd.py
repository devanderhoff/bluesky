from spinup import td3
import tensorflow as tf
import gym
import gym_bluesky
from spinup.utils.test_policy import load_policy, run_policy
import bluesky as bs
from bluesky import tools

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

#
# env = gym.make('bluesky-v0')
# env = DummyVecEnv([lambda: env])
#
# # model = SAC(MlpPolicy, env, verbose=1)
# # model.learn(total_timesteps=50000, log_interval=10)
# # model.save("test2")
#
#
# model = SAC.load("test2")
# #
# # obs = env.reset()
# # while True:
# #     action, _states = model.predict(obs)
# #     obs, rewards, dones, info = env.step(action)
# #     env.render()
#
#
# for i_episode in range(20):
#     obs = env.reset()
#     for t in range(1000):
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         print(action)
#         env.render()
#         if dones:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()




# #
# env = gym.make('bluesky-v0')
#
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(1000):
#         env.render()
#
#         action = env.action_space.sample()
#         print(action)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
# #
# # #
# #


env_fn = lambda : gym.make('bluesky-v0')

ac_kwargs = dict(hidden_sizes=[128,128], activation=tf.nn.tanh)
logger_kwargs = dict(output_dir='/home/dennis/spinupoutput2/run_25', exp_name='run_25')

td3(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=20, logger_kwargs=logger_kwargs, max_ep_len=1000, start_steps=0)


# Run 22 circle
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

