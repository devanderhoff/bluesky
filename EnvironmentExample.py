import gym
import bluesky as bs


class EnvironmentExample(gym.env):
    def __init__(self):
        # Initialize BlueSky
        # With networking
        bs.init()
        # Without networking
        # bs.init(mode='sim-detached')

        # Setup simulation
        bs.sim.connect()

    def step(self, action):
        stackcmd = make_stack_command_from_action(action)

        # Stack command(s) based on ML action
        bs.stack.stack(stackcmd)

        # Perform one simulation timestep
        bs.sim.step()

        state = {
            'lat': bs.traf.lat,
            'lon': bs.traf.lon  # Etc.
        }

        return state

    def reset(self):
        bs.stack.stack('IC IC')

        # Perform one simulation timestep
        bs.sim.step()

        state = {
            'lat': bs.traf.lat,
            'lon': bs.traf.lon  # Etc.
        }

        return state
