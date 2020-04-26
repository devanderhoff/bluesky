""" External control plugin for Machine Learning applications. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, sim, traf, tools  # , net #, navdb, traf, sim, scr, tools
import numpy as np
import collections
import time
from bluesky.traffic import autopilot
from gym import spaces
from ray.rllib.env.policy_client import PolicyClient
from config_ml import Config
from action_dist import BetaDistributionAction, CategoricalOrdinal, CategoricalOrdinalTFP
from model import MyModelCentralized
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_action_dist("CategoricalOrdinalTFP", CategoricalOrdinalTFP)
ModelCatalog.register_custom_model("Centralized", MyModelCentralized)

settings = Config()
settings = settings.load_conf('config_file')

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
relative = True


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.

# Additional initilisation code

def init_plugin():
    # global client_mc
    # client_mc = PolicyClient("http://localhost:27802")

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name': 'MLCONTROLC',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type': 'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 4,

        'update': update,

        'preupdate': preupdate,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        'reset': reset
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
        # Bluesky first timestep starts with update step, so use the reset_bool to determine wheter a reset has
        # occured or not. Then create environment agents.
        if reset_bool:
            # Randomize starting location depending on boundaries in settings.
            aclat = np.random.rand(settings.n_ac) * (settings.max_lat_gen - settings.min_lat_gen) + settings.min_lat_gen
            aclon = np.random.rand(settings.n_ac) * (settings.max_lon_gen - settings.min_lon_gen) + settings.min_lon_gen
            achdg = np.random.randint(1, 360, settings.n_ac)
            acalt = np.ones(settings.n_ac) * 7620
            print('Created ', str(settings.n_ac), ' random aircraft, resetted!')
            traf.create(n=settings.n_ac, aclat=aclat, aclon=aclon, achdg=achdg,  # 360 * np.random.rand(1)
                        acspd=150, acalt=acalt, actype='B777')  # settings.acspd)
            reset_bool = False
            return

        if settings.multiagent:
            # print('Update state:')
            obs, n_ac_neighbours, info = calc_state()
        # else:
        #     print('Singleagent')
        #     obs = calc_state_single()


        reward, done = calc_reward(n_ac_neighbours)
        # print('reward:', reward)
        for agent_id in done.keys():
            if done[agent_id] and not delete_dict[agent_id]:
                traf.delete(traf.id2idx(agent_id))
                print('Deleted ', agent_id)
                done_count += 1
                # final_obs[agent_id] = obs[agent_id]
                delete_dict[agent_id] = True

        # Track nr of timesteps
        idx_mc += 1

        # if idx_mc == settings.max_timesteps or done_count == settings.n_ac:
        #     done['__all__'] = True'

        # Return observations and rewards
        client_mc.log_returns(eid, reward, done, info=[])

        # Check if episode is done either by horizon or all a/c landed, but more likely crashed into each other.
        if idx_mc == settings.max_timesteps or done_count == settings.n_ac:
            # for agent_id in done.keys():
            #     if not done[agent_id]:
            #         final_obs[agent_id] = obs[agent_id]

            # done['__all__'] = True
            # client_mc.log_returns(eid, reward, done, info=[])
            print('Done with Episode: ', eid)
            client_mc.end_episode(eid, obs)
            sim.reset()
            # return

        # client_mc.log_returns(eid, reward, done, info=[])

def preupdate():
    global obs, reset_bool, connected, prev_obs, first_time, final_obs, done, delete_dict, obs_first
    if connected:

        if first_time:
            done = dict.fromkeys(traf.id)
            # done['__all__'] = False
            delete_dict = dict.fromkeys(traf.id, False)
            final_obs = dict.fromkeys(traf.id)
            obs_first = dict.fromkeys(traf.id, None)
            first_time = False
            if settings.multiagent:
                # print('Preupdate obs:')
                obs, n_ac_neighbours, _ = calc_state()
            # else:
            #     obs = calc_state_single()

        # if settings.multiagent:
        #     # print('Preupdate obs:')
        #     obs = calc_state()
        # else:
        #     obs = calc_state_single()

        action = client_mc.get_action(eid, obs)
        # print('Preupdate action NN', action)
        for idx_action, action in action.items():
            # print('Action ', str(action[0]), 'at idx_mc ', idx_mc)
            # print('Before action HDG: ' + str(traf.hdg[0]))
            ## DISCRETE ##
            if action == 0:
                action_hdg = -15
            elif action == 1:
                action_hdg = -10
            elif action == 2:
                action_hdg = -5
            elif action == 3:
                action_hdg = 0
            elif action == 4:
                action_hdg = 5
            elif action == 5:
                action_hdg = 10
            elif action == 6:
                action_hdg = 15

            ## ##
            # print(action)
            # if False:
            #     action_temp = round(action[0] * 180)
            #     action_tot = (obs[idx_action][2] + action_temp) % 360
            # elif True:
            # action_temp = round((action[0] * 40)) - 20
            action_temp = action_hdg
            # action_temp = action*20
            action_tot = (obs[idx_action][2] + action_temp) % 360
            #     # print('action :', action[0], 'totaal :', action_tot)
            # else:
            #     pass
            # print('Preupdate action input', action_tot)
            traf.ap.selhdgcmd(traf.id2idx(idx_action), action_tot)
            # print(action_tot)

        # print(action)
        # action_tot = obs[idx_action][2] + action
        # print(action_tot)
        # traf.ap.selhdgcmd(traf.id2idx(idx_action), action_tot)
        prev_obs = obs

    # stack.stack('HDG ' + traf.id[0] + ' ' + str(action[0]))
    # stack.stack.process()

def reset():
    global reward, idx_mc, eid, reset_bool, connected, first_time, done_count
    # if connected:
    reward = 0
    idx_mc = 0
    done_count = 0
    reset_bool = True
    first_time = True
    client_mc.update_policy_weights()
    eid = client_mc.start_episode(training_enabled=True) # settings.training_enabled)
    connected = True
    print('Resetting with env ID:  ', eid)
    sim.op()
    # traf.trails.setTrails
    sim.fastforward()

def ml_reset(port):
    global client_mc
    host_ip = "http://localhost:" + port
    print("Connected to policy server: ", host_ip)
    client_mc = PolicyClient(host_ip, inference_mode='local', update_interval=None)


def calc_state():
    # Required information:
    # traf.lat
    # traf.lon
    # traf.hdg
    # traf.id
    # latlong eham
    # multi agents obs: lat, long, hdg, dist_wpt, hdg_wpt, dist_plane1, hdg_plane1, dist_plane2, hdg_plane2 (nm/deg)

    # Retrieve latitude and longitude, and add destination lat/long to the end of the list for distance
    # and qdr calculation.

    lat_list = np.append(traf.lat, lat_eham)
    lon_list = np.append(traf.lon, lon_eham)
    traf_id_list = np.tile(traf.id, (len(traf.lon), 1))

    qdr, dist = tools.geo.kwikqdrdist_matrix(np.asmatrix(lat_list), np.asmatrix(lon_list), np.asmatrix(lat_list),
                                             np.asmatrix(lon_list))
    qdr = degto180(qdr)

    # Reshape into; Rows = each aircraft, columns are states.
    obs_matrix_first = np.concatenate([traf.lat.reshape(-1, 1), traf.lon.reshape(-1, 1), traf.hdg.reshape(-1, 1),
                                       np.asarray(dist[:-1, -1]), np.asarray(qdr[:-1, -1])], axis=1)

    # Remove last entry, that is distance calculated to destination.
    dist = np.asarray(dist[:-1, :-1])
    qdr = np.asarray(qdr[:-1, :-1])

    # Sort value's on clostest distance.
    sort_idx = np.argsort(dist, axis=1)
    dist = np.take_along_axis(dist, sort_idx, axis=1)
    qdr = np.take_along_axis(qdr, sort_idx, axis=1)
    traf_id_list_sort = np.take_along_axis(traf_id_list, sort_idx, axis=1)

    # Determine amount of current aircraft and neighbours.
    n_ac_neighbours = np.size(dist, axis=1) - 1
    n_ac_current = np.size(dist, axis=0)

    if n_ac_neighbours > 0 and settings.n_neighbours > 0:

        # Split up values to make [dist, qdr] stacks.
        dist = np.split(dist[:, 1:], np.size(dist[:, 1:], axis=1), axis=1)
        qdr = np.split(qdr[:, 1:], np.size(qdr[:, 1:], axis=1), axis=1)
        traf_id_list_sort = traf_id_list_sort[:, 1:]

        # When current number of neighbours is lower than the amount set in settings, fill rest with dummy variables.
        if n_ac_neighbours < settings.n_neighbours:
            nr_fill = settings.n_neighbours - n_ac_neighbours

            # Dummy values
            fill_mask_dist = np.full((n_ac_current, 1), 0) #settings.max_dist
            fill_mask_qdr = np.full((n_ac_current, 1), 0)
            fill_mask = np.concatenate([fill_mask_dist, fill_mask_qdr], axis=1)
            comb_ac = np.hstack([np.hstack([dist[i], qdr[i]]) for i in range(n_ac_neighbours)])
            for i in range(nr_fill):
                comb_ac = np.hstack((comb_ac, fill_mask))
            n_ac_neighbours_send = n_ac_neighbours

        # If there are still enough neighbouring aircraft, the state space can be made without the use of dummies.
        elif n_ac_neighbours >= settings.n_neighbours and settings.n_neighbours > 0:
            comb_ac = np.hstack([np.hstack([dist[i], qdr[i]]) for i in range(settings.n_neighbours)])
            n_ac_neighbours_send = settings.n_neighbours
            traf_id_list_sort = traf_id_list_sort[:, :settings.n_neighbours]
        # Combine S0 (lat, long, hdg, wpt dist, hdg dist) with other agent information.
        obs_matrix_first = np.concatenate([obs_matrix_first, comb_ac], axis=1)

    elif n_ac_neighbours == 0:

        # Dummy values
        nr_fill = settings.n_neighbours
        fill_mask_dist = np.full((n_ac_current, 1), 0)
        fill_mask_qdr = np.full((n_ac_current, 1), 0)
        fill_mask = np.concatenate([fill_mask_dist, fill_mask_qdr], axis=1)
        for i in range(nr_fill):
            obs_matrix_first = np.concatenate([obs_matrix_first, fill_mask], axis=1)
        n_ac_neighbours_send = n_ac_neighbours
        traf_id_list_sort = np.full((n_ac_current, 1), 0)

    # Create final observation dict, with corresponding traffic ID's.
    obs_c = dict(zip(traf.id, obs_matrix_first))
    info = dict.fromkeys(traf.id, {})
    for idx, keys in enumerate(traf.id):
        info[keys]['sequence'] = list(traf_id_list_sort[idx])
    #
    # info_list = ['sequence'] * len(traf.lon)
    # traf_id_list_c = dict(zip(traf.id, info_list))
    #
    #
    # temp = dict(zip(traf.id, info_list))
    # traf_id_list_c = dict(zip(temp, traf_id_list_sort))

    return obs_c, n_ac_neighbours_send, info

def calc_reward(n_ac_neighbours):
    global obs, prev_obs, done
    # multi agents obs: lat, long, hdg, dist_wpt, hdg_wpt, dist_plane1, hdg_plane1, dist_plane2, hdg_plane2 (nm/deg)
    # This function calculates the "intrensic" reward as well as additional reward shaping

    # Constants to determine faulty states:
    settings.los = 1.05  # nm
    settings.wpt_reached = 5  # nm
    settings.gamma = 0.99  # Match with trainer/implement in settings

    # Create reward dict depending on obs id's.
    reward = dict.fromkeys(obs.keys())

    for agent_id in reward.keys():
        # Initialize reward to 0.
        reward[agent_id] = 0
        done[agent_id] = False

        # First check if goal is reached
        if obs[agent_id][3] <= settings.wpt_reached:
            reward[agent_id] += 1
            done[agent_id] = True

        # Check if there are any collisions
        dist_idx = np.arange(5, 5 + n_ac_neighbours*2 - 1, 2) #len(obs[agent_id])
        if any(obs[agent_id][dist_idx] <= settings.los):
            reward[agent_id] += -1
            done[agent_id] = True

        # Implement reward shaping:
        # F = settings.gamma * (100 / obs[agent_id][3]) - (100 / prev_obs[agent_id][3])

        # Final reward
        # reward[agent_id] += F

    return reward, done


def calc_reward_single():
    global obs, prev_obs, done
    # multi agents obs: lat, long, hdg, dist_wpt, hdg_wpt, dist_plane1, hdg_plane1, dist_plane2, hdg_plane2 (nm/deg)
    ## This function calculates the "intrensic" reward as well as additional reward shaping
    # Constants to determine faulty states:
    settings.los = 1.05  # nm
    settings.wpt_reached = 5  # nm
    settings.gamma = 0.99  # Match with trainer/implement in settings

    reward = dict.fromkeys(obs.keys())

    for agent_id in reward.keys():
        # set initial reward to -1, as for each timestep spend a penalty is introduced.
        reward[agent_id] = 0
        done[agent_id] = False
        # First check if goal is reached
        if obs[agent_id][3] <= settings.wpt_reached:
            reward[agent_id] += 100
            done[agent_id] = True

        # Check if there are any collisions
        dist_idx = np.arange(5, len(obs[agent_id]) - 1, 2)
        if any(obs[agent_id][dist_idx] <= settings.los):
            reward[agent_id] += -200

        # Implement reward shaping:
        F = settings.gamma * (10 / obs[agent_id][3]) - 10 / prev_obs[agent_id][3]

        # Final reward
        reward[agent_id] += F

    return reward, done

def degto180(angle):
    """Change to domain -180,180 """
    return (angle + 180.) % 360 - 180.

# def calc_state_single():
#     global relative, obs_first
#     # Required information:
#     # traf.lat
#     # traf.lon
#     # traf.hdg
#     # traf.id
#     # latlong eham
#     # multi agents obs: lat, long, hdg, dist_wpt, hdg_wpt, dist_plane1, hdg_plane1, dist_plane2, hdg_plane2 (nm/deg)
#     obs_single = obs_first
#
#
#     if relative:
#         idx_list = traf.id2idx(traf.id) + np.ones(np.size(traf.id))
#         idx_list = idx_list.astype(int)
#         # obs_id_first
#         if np.size(traf.id) < settings.n_ac:
#             match_list = np.zeros(np.size(settings.n_ac), dtype=bool)
#             for id_check in traf.id:
#                for match_idx, match_id in enumerate(obs_single.keys()):
#                    if id_check is match_id:
#                        match_list[match_idx] = True
#
#
#
#         lat_list = np.append(lat_eham, traf.lat)
#         lon_list = np.append(lon_eham, traf.lon)
#
#         qdr, dist = tools.geo.kwikqdrdist_matrix(np.asmatrix(lat_list), np.asmatrix(lon_list), np.asmatrix(lat_list),
#                                                  np.asmatrix(lon_list))
#         qdr = np.asarray(qdr)
#         qdr = degto180(qdr)
#         dist = np.asarray(dist)
#
#         n_ac_current = np.size(dist, axis=0) - 1
#
#         for i, trafID in enumerate(traf.id):
#             # qdr and dist to eham.
#             qdr_s0 = qdr[0, idx_list[i]]
#             dist_s0 = dist[0, idx_list[i]]
#             s0_comb = np.hstack((qdr_s0, dist_s0))
#
#             dist_plane = np.hstack((dist[idx_list[i], :idx_list[i]], dist[idx_list[i], idx_list[i]+1:]))
#             qdr_plane = np.hstack((qdr[idx_list[i], :idx_list[i]], dist[idx_list[i], idx_list[i]+1:]))
#
#             dist_plane = np.split(dist_plane, n_ac_current)
#             qdr_plane = np.split(qdr_plane, n_ac_current)
#
#             nr_fill = settings.n_ac - n_ac_current
#             fill_mask_dist = np.array([settings.max_dist])
#             fill_mask_qdr = np.array([180])
#             fill_mask = np.concatenate([fill_mask_dist, fill_mask_qdr])
#
#             comb_ac = np.hstack([np.hstack([qdr_plane[i], dist_plane[i]]) for i in range(n_ac_current)])
#
#             if nr_fill > 0:
#                 for _ in range(nr_fill):
#                     comb_ac = np.hstack((comb_ac, fill_mask))
#
#             comb_ac = np.hstack((s0_comb, comb_ac))
#             obs_single[trafID] = comb_ac
#
#         for value, key in enumerate(obs_single):
#             if value is None:
#                 fill_mask_dist = np.array([settings.max_dist])
#                 fill_mask_qdr = np.array([180])
#                 comb_fill = np.concatenate([fill_mask_dist, fill_mask_qdr])
#                 for i in range(settings.n_ac):
#                     comb_fill = np.hstack((comb_fill, comb_fill))
#                 obs_single[key] = comb_fill
#         print('loop')
#     return obs_single