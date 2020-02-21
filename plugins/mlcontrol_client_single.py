""" External control plugin for Machine Learning applications. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, sim, traf, settings, tools #, net #, navdb, traf, sim, scr, tools
import numpy as np
import collections
import time
from bluesky.traffic import autopilot
from gym import spaces
from ray.rllib.utils.policy_client import PolicyClient

settings.set_variable_defaults(n_ac=1, training_enabled=True, acspd=250, nr_nodes=4, min_lat = 51 , max_lat = 54, min_lon =2 , max_lon = 8, n_neighbours=2 )
print(settings.n_ac)
client_mc = None
reward = None
idx_mc = None
reset_bool = True
eid = None
obs = None
prev_obs = None
connected = False
first_time = None
done = None
delete_dict = None
done_count = 0
lat_eham = 52.18
lon_eham = 4.45
### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.

# Additional initilisation code

def init_plugin():
    # global client_mc
    # client_mc = PolicyClient("http://localhost:27802")

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'MLCONTROLCSINGLE',

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
            'MLRESET port',

            # A list of the argument types your function accepts. For a description of this, see ...
            'txt',

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
    global reward, idx_mc, reset_bool, connected, obs, prev_obs, done_count, final_obs, done, delete_dict
    if connected:
        # Bluesky first timestep starts with update step, so use the reset_bool to determine wheter a reset has occured or not. Then create environment agents.
        if reset_bool:
            #Randomize starting location depending on boundaries in settings.
            aclat = np.random.rand(settings.n_ac) * (settings.max_lat - settings.min_lat) + settings.min_lat
            aclon = np.random.rand(settings.n_ac) * (settings.max_lon - settings.min_lon) + settings.min_lon
            achdg = np.random.randint(1, 360, settings.n_ac)
            acalt = np.ones(settings.n_ac) * 7620
            # acspd = np.ones(settings.n_ac) * 300
            print('Created ', str(settings.n_ac), ' random aircraft, resetted!')
            traf.create(n=settings.n_ac, aclat=aclat, aclon=aclon, achdg=achdg, #360 * np.random.rand(1)
            acspd=150, acalt=acalt, actype='B777')#settings.acspd)
            reset_bool = False
            return
        # print('After action HDG: ' + str(traf.hdg[0]))
        obs = calc_state()
        reward, done = calc_reward()

        for agent_id in done.keys():
            if done[agent_id] and not delete_dict[agent_id]:
                traf.delete(traf.id2idx(agent_id))
                print('Deleted ', agent_id)
                done_count += 1
                final_obs[agent_id] = obs[agent_id]
                delete_dict[agent_id] = True

        idx_mc += 1
        client_mc.log_returns(eid, reward, done, info=[])
        if idx_mc == 500 or done_count == settings.n_ac:
            for agent_id in done.keys():
                if not done[agent_id]:
                    final_obs[agent_id] = obs[agent_id]

            print('total reward', reward)
            print('Done with Episode: ', eid)
            client_mc.end_episode(eid, final_obs)
            sim.reset()

def preupdate():
    global obs, reset_bool, connected, prev_obs, first_time, final_obs, done, delete_dict
    if connected:
        obs = calc_state()
        if first_time:
            done = dict.fromkeys(obs.keys())
            delete_dict = dict.fromkeys(obs.keys(), False)
            final_obs = dict.fromkeys(obs.keys())
            first_time = False
        action = client_mc.get_action(eid, obs)
        # print(action)
        for idx_mc, action in action.items():
            # print('Action ', str(action[0]), 'at idx_mc ', idx_mc)
            # print('Before action HDG: ' + str(traf.hdg[0]))
            if action == 0:
                action_hdg = -25
            elif action == 1:
                action_hdg = -12
            elif action == 2:
                action_hdg = 0
            elif action == 3:
                action_hdg = 12
            else:
                action_hdg = 25

            action_tot = obs[idx_mc][2] + action_hdg
            traf.ap.selhdgcmd(traf.id2idx(idx_mc), action_tot)
        prev_obs = obs



    # stack.stack('HDG ' + traf.id[0] + ' ' + str(action[0]))
    # stack.stack.process()




def reset():
    global reward, idx_mc, eid, reset_bool, connected, first_time
    reward = 0
    idx_mc = 0
    reset_bool = True
    first_time = True
    eid = client_mc.start_episode(training_enabled=settings.training_enabled)
    connected = True
    print('Resetting with env ID:  ', eid)
    sim.op()
    # traf.trails.setTrails
    sim.fastforward()

def ml_reset(port):
    global client_mc
    host_ip = "http://localhost:" + port
    print(host_ip)
    client_mc = PolicyClient(host_ip)


def calc_state():
    # Required information:
    #traf.lat
    #traf.lon
    #traf.hdg
    #traf.id
    #latlong eham
    # multi agents obs: lat, long, hdg, dist_wpt, hdg_wpt, dist_plane1, hdg_plane1, dist_plane2, hdg_plane2 (nm/deg)
    lat_list = np.append(traf.lat, lat_eham)
    lon_list = np.append(traf.lon, lon_eham)

    qdr, dist = tools.geo.kwikqdrdist_matrix(np.asmatrix(lat_list), np.asmatrix(lon_list), np.asmatrix(lat_list),
                                             np.asmatrix(lon_list))

    obs_matrix_first = np.concatenate([traf.lat.reshape(-1, 1), traf.lon.reshape(-1, 1), traf.hdg.reshape(-1, 1),
                                       np.asarray(dist[:-1, -1]), np.asarray(qdr[:-1, -1])], axis=1)

    dist = np.asarray(dist[:-1, :-1])
    qdr = np.asarray(qdr[:-1, :-1])

    sort_idx = np.argsort(dist, axis=1)
    dist = np.take_along_axis(dist, sort_idx, axis=1)
    qdr = np.take_along_axis(qdr, sort_idx, axis=1)

    n_ac_neighbours = np.size(dist, axis=1) - 1
    n_ac_current = np.size(dist, axis=0)

    dist = np.split(dist[:, 1:], np.size(dist[:, 1:], axis=1), axis=1)
    qdr = np.split(qdr[:, 1:], np.size(qdr[:, 1:], axis=1), axis=1)



    if n_ac_neighbours < settings.n_neighbours:
        nr_fill = settings.n_neighbours - n_ac_neighbours
        fill_mask_dist = np.full((n_ac_current,1), 999)
        fill_mask_qdr = np.full((n_ac_current,1), 180)
        fill_mask = np.concatenate([fill_mask_dist,fill_mask_qdr], axis=1)
        comb_ac = np.hstack([np.hstack([dist[i], qdr[i]])for i in range(n_ac_neighbours)])
        for i in range(nr_fill):
            comb_ac = np.hstack((comb_ac, fill_mask))
    else:
        comb_ac = np.hstack([np.hstack([dist[i], qdr[i]]) for i in range(settings.n_neighbours)])

    obs_matrix_first = np.concatenate([obs_matrix_first, comb_ac], axis=1)
    obs_c = dict(zip(traf.id, obs_matrix_first))

    return obs_c


def calc_reward():
    global obs, prev_obs, done
    # multi agents obs: lat, long, hdg, dist_wpt, hdg_wpt, dist_plane1, hdg_plane1, dist_plane2, hdg_plane2 (nm/deg)
    ## This function calculates the "intrensic" reward as well as additional reward shaping
    # Constants to determine faulty states:
    los = 1.05 #nm
    wpt_reached = 5 #nm
    gamma = 0.99 #Match with trainer/implement in settings

    reward = dict.fromkeys(obs.keys())


    for agent_id in reward.keys():
        # set initial reward to -1, as for each timestep spend a penalty is introduced.
        reward[agent_id] = 0
        done[agent_id] = False
        # First check if goal is reached
        if obs[agent_id][3] <= wpt_reached:
            reward[agent_id] += 100
            done[agent_id] = True

        # Check if there are any collisions
        dist_idx = np.arange(5, len(obs[agent_id])-1, 2)
        if any(obs[agent_id][dist_idx] <= los):
            reward[agent_id] += -200

        # Implement reward shaping:
        F = gamma*(10/obs[agent_id][3]) - 10/prev_obs[agent_id][3]

        # Final reward
        reward[agent_id] += F

    return reward, done

def rand_latlon():
    aclat = np.random.rand(settings.n_ac) * (settings.max_lat - settings.min_lat) + settings.min_lat
    aclon = np.random.rand(settings.n_ac) * (settings.max_lon - settings.min_lon) + settings.min_lon
    achdg = np.random.randint(1, 360, settings.n_ac)
    acspd = np.ones(settings.n_ac) * 250