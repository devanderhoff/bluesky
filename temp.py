from bluesky_env_ray import BlueSkyEnv

class EnvConfigBuild:
    def __init__(self, worker_index, vector_index):
        self.worker_index = worker_index
        self.vector_index = vector_index


env_config = EnvConfigBuild(1, 1)


env = BlueSkyEnv(env_config)


env.reset()

for i in range(100):
    env.step(60)


