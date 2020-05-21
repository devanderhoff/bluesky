""" External control plugin for Machine Learning applications. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, sim, traf, tools, navdb  # , net #, navdb, traf, sim, scr, tools
import numpy as np
import collections
import time
from bluesky.traffic import autopilot
from gym import spaces
from ray.rllib.env.policy_client import PolicyClient
from config_ml import Config
from action_dist import BetaDistributionAction, CategoricalOrdinal, CategoricalOrdinalTFP
from model import MyModelCentralized
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
# from Centralized import centralized_critic_postprocessing, loss_with_central_critic, setup_mixins, central_vf_stats,\
#     LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, CentralizedValueMixin
import Centralized

ModelCatalog.register_custom_action_dist("CategoricalOrdinalTFP", CategoricalOrdinalTFP)
ModelCatalog.register_custom_model("Centralized", MyModelCentralized)

# CCPPO = PPOTFPolicy.with_updates(
#     name="CCPPO",
#     postprocess_fn=centralized_critic_postprocessing,
#     loss_fn=loss_with_central_critic,
#     before_loss_init=setup_mixins,
#     grad_stats_fn=central_vf_stats,
#     mixins=[
#         LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
#         CentralizedValueMixin
#     ])
#
# CCTrainer = PPOTrainer.with_updates(name="CCPPOTrainer", default_policy=CCPPO)

settings = Config()
settings = settings.load_conf('config_file')

destination = None
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
            aclat, aclon = latlon_randomizer(settings)
            achdg = np.random.randint(1, 360, settings.n_ac)
            acalt = np.ones(settings.n_ac) * 7620
            dest_idx = np.random.randint(3, size=settings.n_ac) + 1
            print('Created ', str(settings.n_ac), ' random aircraft, resetted!')
            traf.create(n=settings.n_ac, aclat=aclat, aclon=aclon, achdg=achdg,  # 360 * np.random.rand(1)
                        acspd=150, acalt=acalt, actype='B777', dest=dest_idx)  # settings.acspd)
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
        client_mc.log_returns(eid, reward, info, done)

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
    global obs, reset_bool, connected, prev_obs, first_time, final_obs, done, delete_dict, obs_first, dest_idx
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
            action_tot = (obs[idx_action][3] + action_temp) % 360
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
    # multi agents obs: destination ID, lat, long, hdg, dist_wpt, hdg_wpt, dist_plane1, hdg_plane1, dist_plane2, hdg_plane2 (nm/deg)

    # Determine lat long for destination set
    # -3 = EHAM
    # -2 = EHEH
    # -1 = EHDL
    dest_idx_list = [navdb.getaptidx('EHAM'), navdb.getaptidx('EHEH'), navdb.getaptidx('EHDL')]
    lat_dest_list = navdb.aptlat[dest_idx_list]
    lon_dest_list = navdb.aptlon[dest_idx_list]



    # Retrieve latitude and longitude, and add destination lat/long to the end of the list for distance
    # and qdr calculation.

    # lat_list = np.append(traf.lat, lat_eham)
    # lon_list = np.append(traf.lon, lon_eham)
    lat_list = np.append(traf.lat, lat_dest_list)
    lon_list = np.append(traf.lon, lon_dest_list)
    traf_id_list = np.tile(traf.id, (len(traf.lon), 1))


    qdr, dist = tools.geo.kwikqdrdist_matrix(np.asmatrix(lat_list), np.asmatrix(lon_list), np.asmatrix(lat_list),
                                             np.asmatrix(lon_list))
    qdr = degto180(qdr)

    traf_idx = np.arange(len(traf.id))
    dist_destination = dist[traf_idx, -traf.dest_temp]
    qdr_destination = qdr[traf_idx, -traf.dest_temp]

    # # Reshape into; Rows = each aircraft, columns are states. OLD
    # obs_matrix_first = np.concatenate([traf.lat.reshape(-1, 1), traf.lon.reshape(-1, 1), traf.hdg.reshape(-1, 1),
    #                                    np.asarray(dist[:-1, -1]), np.asarray(qdr[:-1, -1])], axis=1)

    # Reshape into; Rows = each aircraft, columns are states
    obs_matrix_first = np.concatenate([traf.dest_temp.reshape(-1, 1), traf.lat.reshape(-1, 1), traf.lon.reshape(-1, 1),
                                       traf.hdg.reshape(-1, 1), np.asarray(dist_destination.reshape(-1, 1)),
                                       np.asarray(qdr_destination.reshape(-1,1))], axis=1)

    # Remove last entry, that is distance calculated to destination.
    dist = np.asarray(dist[:-3, :-3])
    qdr = np.asarray(qdr[:-3, :-3])

    # Sort value's on clostest distance.
    sort_idx = np.argsort(dist, axis=1)
    dist = np.take_along_axis(dist, sort_idx, axis=1)
    qdr = np.take_along_axis(qdr, sort_idx, axis=1)
    traf_id_list_sort = np.take_along_axis(traf_id_list, sort_idx, axis=1)
    traf_send_flag = True

    # Determine amount of current aircraft and neighbours.
    n_ac_neighbours = np.size(dist, axis=1) - 1
    n_ac_current = np.size(dist, axis=0)

    if n_ac_neighbours > 0 and settings.n_neighbours > 0:

        # Split up values to make [dist, qdr] stacks.
        dist = np.split(dist[:, 1:], np.size(dist[:, 1:], axis=1), axis=1)
        qdr = np.split(qdr[:, 1:], np.size(qdr[:, 1:], axis=1), axis=1)
        # traf_id_list_sort = traf_id_list_sort[:, 1:]

        # When current number of neighbours is lower than the amount set in settings, fill rest with dummy variables.
        if n_ac_neighbours < settings.n_neighbours:
            nr_fill = settings.n_neighbours - n_ac_neighbours

            # Dummy values
            fill_mask_dist = np.full((n_ac_current, 1), -1) #settings.max_dist
            fill_mask_qdr = np.full((n_ac_current, 1), -1)
            fill_mask = np.concatenate([fill_mask_dist, fill_mask_qdr], axis=1)
            comb_ac = np.hstack([np.hstack([dist[i], qdr[i]]) for i in range(n_ac_neighbours)])
            for i in range(nr_fill):
                comb_ac = np.hstack((comb_ac, fill_mask))
            n_ac_neighbours_send = n_ac_neighbours

        # If there are still enough neighbouring aircraft, the state space can be made without the use of dummies.
        elif n_ac_neighbours >= settings.n_neighbours and settings.n_neighbours > 0:
            comb_ac = np.hstack([np.hstack([dist[i], qdr[i]]) for i in range(settings.n_neighbours)])
            n_ac_neighbours_send = settings.n_neighbours
            traf_id_list_sort = traf_id_list_sort[:, :settings.n_neighbours+1]
        # Combine S0 (lat, long, hdg, wpt dist, hdg dist) with other agent information.
        obs_matrix_first = np.concatenate([obs_matrix_first, comb_ac], axis=1)

    elif n_ac_neighbours == 0:

        # Dummy values
        nr_fill = settings.n_neighbours
        fill_mask_dist = np.full((n_ac_current, 1), -1)
        fill_mask_qdr = np.full((n_ac_current, 1), -1)
        fill_mask = np.concatenate([fill_mask_dist, fill_mask_qdr], axis=1)
        for i in range(nr_fill):
            obs_matrix_first = np.concatenate([obs_matrix_first, fill_mask], axis=1)
        n_ac_neighbours_send = n_ac_neighbours
        traf_id_list_sort = []
        traf_send_flag = False

    if settings.n_neighbours == 0:
        n_ac_neighbours_send = settings.n_neighbours
        traf_id_list_sort = []
        traf_send_flag = False
    # Create final observation dict, with corresponding traffic ID's.
    obs_c = dict(zip(traf.id, obs_matrix_first))
    info = dict.fromkeys(traf.id, {})
    # Remove check later.
    if traf_send_flag:
        for idx, keys in enumerate(traf.id):
            if keys == traf_id_list_sort[idx, 0]:
                info[keys]= {'sequence': list(traf_id_list_sort[idx])}
            else:
                raise ValueError("Info dict is wrongly constructed")
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
    # settings.los = 1.05  # nm
    # settings.wpt_reached = 5  # nm
    settings.gamma = 0.99  # Match with trainer/implement in settings

    # Create reward dict depending on obs id's.
    reward = dict.fromkeys(obs.keys())

    for agent_id in reward.keys():
        # Initialize reward to 0.
        reward[agent_id] = 0
        done[agent_id] = False

        # First check if goal is reached
        if obs[agent_id][4] <= settings.wpt_reached:
            reward[agent_id] += 1
            done[agent_id] = True

        # Check if there are any collisions
        dist_idx = np.arange(6, 6 + n_ac_neighbours*2 - 1, 2) #len(obs[agent_id])
        if any(obs[agent_id][dist_idx] <= settings.los):
            reward[agent_id] += -1
            done[agent_id] = True

        # Implement reward shaping:
        # F = settings.gamma * (100 / obs[agent_id][3]) - (100 / prev_obs[agent_id][3])

        # Final reward
        # reward[agent_id] += F

    return reward, done

def degto180(angle):
    """Change to domain -180,180 """
    return (angle + 180.) % 360 - 180.

def latlon_randomizer(settings):
    done_flag = False
    iter_n = 0
    while not done_flag:
        iter_n += 1
        aclat = np.random.rand(settings.n_ac) * (settings.max_lat_gen - settings.min_lat_gen) + settings.min_lat_gen
        aclon = np.random.rand(settings.n_ac) * (settings.max_lon_gen - settings.min_lon_gen) + settings.min_lon_gen
        _, dist = tools.geo.kwikqdrdist_matrix(np.asmatrix(aclat), np.asmatrix(aclon), np.asmatrix(aclat),
                                               np.asmatrix(aclon))
        mask = np.mask_indices(settings.n_ac, np.tril, -1)
        dist = np.asarray(dist)
        dist = dist[mask]
        if all(dist >= settings.spawn_separation) or iter_n >= 1000:
            done_flag = True
            if iter_n >= 1000:
                print("Maximum iterations on aircraft random generation reached, aircraft may spawn to close together")
                print("Minimum current distance is: ", min(dist))
    return aclat, aclon

