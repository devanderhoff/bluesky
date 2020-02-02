""" External control plugin for Machine Learning applications. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, sim, traf, settings, tools #, net #, navdb, traf, sim, scr, tools
import numpy as np
import time
from bluesky.traffic import autopilot
from gym import spaces
from ray.rllib.utils.policy_client import PolicyClient

settings.set_variable_defaults(n_ac=1, training_enabled=True, acspd=250, nr_nodes=4, min_lat = 51 , max_lat = 54, min_lon =2 , max_lon = 8 )

client_mc = None
reward = None
idx_mc = None
reset_bool = True
eid = None
obs = None
connected = False
lat_eham = 52.18
lon_eham = 4.45
### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.

# Additional initilisation code

def init_plugin():
    global client_mc
    client_mc = PolicyClient("http://localhost:27802")

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

        'preupdate':       preupdate,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        'reset':         reset
        }

    stackfunctions = {
        # The command name for your function
        'MLRESET': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'MLSTEP',

            # A list of the argument types your function accepts. For a description of this, see ...
            '',

            # The name of your function in this plugin
            ml_reset,

            # a longer help text of your function.
            'Simulate one MLCONTROL time interval.']
    }
    # Set constants for environment

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

def update():
    global reward, idx_mc, reset_bool, connected
    if connected:
        # Bluesky first timestep starts with update step, so use the reset_bool to determine wheter a reset has occured or not. Then create environment agents.
        if reset_bool:
            #Randomize starting location depending on boundaries in settings.
            aclat = np.random.rand(settings.n_ac) * (settings.max_lat - settings.min_lat) + settings.min_lat
            aclon = np.random.rand(settings.n_ac) * (settings.max_lon - settings.min_lon) + settings.min_lon
            achdg = np.random.randint(1, 360, settings.n_ac)

            print('Created ', str(settings.n_ac), ' random aircraft, resetted!')
            traf.create(n=settings.n_ac, aclat=aclat, aclon=aclon, achdg=achdg, #360 * np.random.rand(1)
            acspd=300)#settings.acspd)
            reset_bool = False
            return
        # print('After action HDG: ' + str(traf.hdg[0]))
        # calc_state()
        dist_wpt = tools.geo.kwikdist(traf.lat[0], traf.lon[0], lat_eham, lon_eham)
        # print(dist_wpt)
        reward += 3/dist_wpt
        idx_mc += 1
        client_mc.log_returns(eid, reward, info=[])
        if idx_mc == 500:
            print('total reward', reward)
            print('Done with Episode: ', eid)
            client_mc.end_episode(eid, obs)
            sim.reset()

def preupdate():
    global obs, reset_bool, connected
    if connected:

        dist_wpt = tools.geo.kwikdist(traf.lat[0], traf.lon[0], lat_eham, lon_eham)


        obs = [traf.lat[0], traf.lon[0], traf.hdg[0], dist_wpt]


        action = client_mc.get_action(eid, obs)

        # print('Action ', str(action[0]), 'at idx_mc ', idx_mc)
        # print('Before action HDG: ' + str(traf.hdg[0]))
        traf.ap.selhdgcmd(traf.id2idx(traf.id[0]), action[0])


    # stack.stack('HDG ' + traf.id[0] + ' ' + str(action[0]))
    # stack.stack.process()




def reset():
    global reward, idx_mc, eid, reset_bool, connected
    reward = 0
    idx_mc = 0
    reset_bool = True
    eid = client_mc.start_episode(training_enabled=settings.training_enabled)
    connected = True
    print('Resetting with env ID:  ', eid)
    sim.op()
    sim.fastforward()

def ml_reset():
    global reward, idx_mc, eid, reset_bool, action_count
    reward = 0
    idx_mc = 0
    reset_bool = True
    action_count = 0
    eid = client_mc.start_episode(training_enabled=settings.training_enabled)
    print('reset: ', eid)
    sim.op()


def calc_state():
    # Required information:
    #traf.lat
    #traf.lon
    #traf.hdg
    #traf.id
    #latlong eham
    lat_list = np.append(traf.lat, lat_eham)
    lon_list = np.append(traf.lon, lon_eham)

    qdr, dist = tools.geo.kwikqdrdist_matrix(traf.lat, traf.lon, traf.lat, traf.lon)
    print('help')

    return

def rand_latlon():
    aclat = np.random.rand(settings.n_ac) * (settings.max_lat - settings.min_lat) + settings.min_lat
    aclon = np.random.rand(settings.n_ac) * (settings.max_lon - settings.min_lon) + settings.min_lon
    achdg = np.random.randint(1, 360, settings.n_ac)
    acspd = np.ones(settings.n_ac) * 250