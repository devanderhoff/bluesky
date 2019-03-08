from gym.envs.registration import register

register(
    id='bluesky-v0',
    entry_point='gym_bluesky.envs:BlueSkyEnv',
)
