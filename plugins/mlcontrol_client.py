""" External control plugin for Machine Learning applications. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, sim, traf  #, settings, navdb, traf, sim, scr, tools
import numpy as np
from gym import spaces
from bluesky import tools
from ray.rllib.utils.policy_client import PolicyClient

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code
    global client_mc
    global iter
    global connected
    connected = False
    client_mc = PolicyClient("https://localhost:27802")
    print(client_mc)
    print('penis')
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'MLCONTROLC',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        # 'update_interval': 1,

        'update':          update,

        # 'preupdate':       preupdate,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        'reset':         reset
        }

    stackfunctions = {
        # The command name for your function
        'MLSTEP': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'MLSTEP',

            # A list of the argument types your function accepts. For a description of this, see ...
            '',

            # The name of your function in this plugin
            mlstep,

            # a longer help text of your function.
            'Simulate one MLCONTROL time interval.']
    }
    # Set constants for environment

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

def update():
    global connected
    global client_mc


    if connected == False:
        eid = client_mc.start_episode(training_enabled=True)
        connected = True
    obs = [traf.lat, traf.lon, traf.hdg]
    action = client_mc.get_action(eid, obs)
    traf.selhdgcmd(traf.id, action)
    reward += 1
    iter += 1
    client_mc.log_returns(eid, reward, info=[])
    if iter ==500:
        print('total reward', reward)
        client_mc.end_episode(eid, obs)
        connected = False

def preupdate():
    pass

def reset():
    global eid
    global iter
    global reward
    global client_mc
    eid = client_mc.start_episode(training_enabled=True)
    traf.create(n=1, aclat=52, aclon=6, achdg=np.random.randn(360), acspd=250)
    reward = 0
    iter = 0


def mlstep():
    pass

