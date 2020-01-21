# policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[128,128, dict(vf=[128,128], pi=[128,128])])
# model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='/home/dennis/tensorboard/test8', n_steps=500, learning_rate=0.003, vf_coef= 0.8, noptepochs=6, nminibatches=16, full_tensorboard_log=True, policy_kwargs=policy_kwargs,ent_coef=0.01)


import numpy as np
