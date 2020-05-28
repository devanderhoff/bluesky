""" External control plugin for Machine Learning applications. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import sim, traf, tools, navdb, net, scr #, navdb, traf, sim, scr, tools
from bluesky.network import client
import numpy as np
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
import time
ModelCatalog.register_custom_action_dist("CategoricalOrdinalTFP", CategoricalOrdinalTFP)
ModelCatalog.register_custom_model("Centralized", MyModelCentralized)

settings = Config()
settings = settings.load_conf('config_file')

obs = None
client_mc = None
idx_mc = None
reset_bool = True
eid = None
connected = False
first_time = None
done = None
delete_dict = None
done_count = 0
# landed_count = 0
# crashed_count = 0
# semi_landed_count = 0
lat_eham = 52.18
lon_eham = 4.45

host_ip = "http://localhost:27800"
print("Connected to policy server: ", host_ip)
client_mc = PolicyClient(host_ip, inference_mode='local', update_interval=None)

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
        ,
        'START_EXPERIMENT': [
            'START_EXPERIMENT ON/OFF',
            '[onoff]',
            start_experiment,
            'simulate'
        ]
    }
    # Set constants for environment

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

def update():
    global connected, reset_bool, done, done_count, delete_dict, idx_mc, obs
    # global reward, idx_mc, reset_bool, connected, obs, prev_obs, done_count, final_obs, done, delete_dict
    if connected:
        # Bluesky first timestep starts with update step, so use the reset_bool to determine wheter a reset has
        # occured or not. Then create environment agents.
        if reset_bool:
            # Randomize starting location depending on boundaries in settings.
            aclat, aclon = latlon_randomizer(settings)
            achdg = np.random.randint(1, 360, settings.n_ac)
            acalt = np.ones(settings.n_ac) * 7620

            # Create equal distribution of destinations
            dest_idx, dest_hdg = destination_creator(settings)
            # print('Heading destination', dest_hdg)
            # print('Destination idx', dest_idx)

            # dest_idx = np.random.randint(3, size=settings.n_ac) + 1
            print('Created ', str(settings.n_ac), ' random aircraft, resetted!')
            traf.create(n=settings.n_ac, aclat=aclat, aclon=aclon, achdg=achdg,  # 360 * np.random.rand(1)
                        acspd=150, acalt=acalt, actype='B777', dest=dest_idx, dest_hdg=dest_hdg)  # settings.acspd)
            reset_bool = False
            return

        # Calculate observations, current amount of n_ac_neighbours and info dict. Info dict is used in the centralized
        # variant to correctly insert other agent actions.
        obs, n_ac_neighbours, info = calc_state()

        # Calculate reward depending on current obs, and check if episode/agents are done.
        reward, done = calc_reward(n_ac_neighbours)

        # Delete agents that are set to done.
        for agent_id in done.keys():
            if done[agent_id] and not delete_dict[agent_id]:
                traf.delete(traf.id2idx(agent_id))
                print('Deleted ', agent_id)
                done_count += 1
                delete_dict[agent_id] = True

        # Track nr of timesteps
        idx_mc += 1

        # Return observations and rewards
        client_mc.log_returns(eid, reward, info, done)

        # Check if episode is done either by horizon or all a/c landed, but more likely crashed into each other.
        if idx_mc == settings.max_timesteps or done_count == (settings.n_ac - settings.n_neighbours + 1):
            print('Done with Episode: ', eid)
            client_mc.end_episode(eid, obs)
            sim.reset()

def preupdate():
    global connected, first_time, done, delete_dict, obs
    # global obs, reset_bool, connected, prev_obs, first_time, final_obs, done, delete_dict, obs_first, dest_idx
    if connected:
        if first_time:

            for n, id in enumerate(traf.id):
                if traf.dest_temp[n] == 1:
                    scr.color(id, 51, 255, 255)
                elif traf.dest_temp[n] == 2:
                    scr.color(id, 255, 51, 51)
                elif traf.dest_temp[n] == 3:
                    scr.color(id, 255,153,51)

            done = dict.fromkeys(traf.id)
            delete_dict = dict.fromkeys(traf.id, False)
            first_time = False
            obs, n_ac_neighbours, _ = calc_state()

        action = client_mc.get_action(eid, obs)

        for idx_action, action in action.items():
            # Define discrete actions. Return from the policy is 0-6 (7 actions total).
            # Define standard action
            action_hdg = 0

            # Convert action to relative heading change.
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

            # Add selected relative heading to current heading, and send as action command.
            action_tot = (obs[idx_action][4] + action_hdg) % 360
            traf.ap.selhdgcmd(traf.id2idx(idx_action), action_tot)

def reset():
    global idx_mc, done_count, reset_bool, first_time, eid, connected
    # if connected:
    idx_mc = 0
    done_count = 0
    # landed_count = 0
    # semi_landed_count = 0
    # crashed_count = 0

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

def start_experiment(flag=True):
    print(flag)
    net.send_event(b'TEST')

def calc_state():
    # Required information:
    # traf.lat
    # traf.lon
    # traf.hdg
    # traf.id
    # latlong eham
    # multi agents obs: destination ID, lat, long, hdg, dist_wpt, hdg_wpt, dist_plane1, hdg_plane1, dist_plane2, hdg_plane2 (nm/deg)

    # Determine lat long for destination set
    # -3 = EHDL
    # -2 = EHEH
    # -1 = EHAM

    if settings.multi_destination:
        dest_idx_list = [navdb.getaptidx('EHDL'), navdb.getaptidx('EHEH'), navdb.getaptidx('EHAM')]
        lat_dest_list = navdb.aptlat[dest_idx_list]
        lon_dest_list = navdb.aptlon[dest_idx_list]
    else:
        dest_idx_list = navdb.getaptidx('EHAM')
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
    obs_matrix_first = np.concatenate([traf.dest_temp.reshape(-1, 1), traf.dest_hdg.reshape(-1, 1), traf.lat.reshape(-1, 1), traf.lon.reshape(-1, 1),
                                       traf.hdg.reshape(-1, 1), np.asarray(dist_destination.reshape(-1, 1)),
                                       np.asarray(qdr_destination.reshape(-1, 1))], axis=1)

    # Remove last entry, that is distance calculated to destination.
    if settings.multi_destination:
        dist = np.asarray(dist[:-3, :-3])
        qdr = np.asarray(qdr[:-3, :-3])
    else:
        dist = np.asarray(dist[:-1, :-1])
        qdr = np.asarray(qdr[:-1, :-1])

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
                info[keys] = {'sequence': list(traf_id_list_sort[idx])}
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
    global done, obs
    # multi agents obs: lat, long, hdg, dist_wpt, hdg_wpt, dist_plane1, hdg_plane1, dist_plane2, hdg_plane2 (nm/deg)
    # This function calculates the "intrensic" reward as well as additional reward shaping

    # Constants to determine faulty states:
    # settings.los = 1.05  # nm
    # settings.wpt_reached = 5  # nm
    settings.gamma = 0.99  # Match with trainer/implement in settings

    landed_bool = False
    semi_landed_bool = False
    crashed_bool = False

    # Create reward dict depending on obs id's.
    reward = dict.fromkeys(obs.keys())

    for agent_id in reward.keys():
        # Initialize reward to 0.
        reward[agent_id] = 0
        done[agent_id] = False
        # First check if goal is reached
        if obs[agent_id][5] <= settings.wpt_reached:
            if settings.destination_hdg:
                # Determine approach hdg for destinations if wanted.
                # 1. EHAM: [1: 340-20]
                #       [2: 70-110]
                #       [3: 210-250]
                #
                # 2. EHGR/EHEH: [1: 260-300]
                #       [2: 190-230]
                #
                # 3. EHDL: [1: 190-230]
                #       [2: 10-40 ]
                correct_hdg = check_hdg_constraint(obs[agent_id][0], obs[agent_id][1], obs[agent_id][4])
                if correct_hdg:
                    reward[agent_id] += 2
                    done[agent_id] = True
                    landed_bool = True
                else:
                    reward[agent_id] += 1
                    done[agent_id] = True
                    semi_landed_bool = True
            else:
                reward[agent_id] += 1
                done[agent_id] = True
                landed_bool = True

        # Check if there are any collisions
        dist_idx = np.arange(7, 7 + n_ac_neighbours*2 - 1, 2) #len(obs[agent_id])
        if any(obs[agent_id][dist_idx] <= settings.los):
            reward[agent_id] += -1
            done[agent_id] = True
            crashed_bool = True

        # Implement reward shaping:
        # F = settings.gamma * (100 / obs[agent_id][3]) - (100 / prev_obs[agent_id][3])

        # Final reward
        # reward[agent_id] += F
        # temp = info[agent_id]['sequence']
        # info[agent_id] = {'sequence': temp, 'metrics': [landed_bool, semi_landed_bool, crashed_bool]}
    return reward, done

def check_hdg_constraint(dest_idx, hdg_idx, cur_hdg):
    # 1. EHAM: [1: 340-20]
    #       [2: 70-110]
    #       [3: 210-250]
    #
    # 2. EHGR/EHEH: [1: 260-300]
    #       [2: 190-230]
    #
    # 3. EHDL: [1: 190-230]
    #       [2: 10-50 ]
    good_approach = False
    if dest_idx == 1:
        if hdg_idx == 1:
            good_approach = (340 <= cur_hdg <= 360 or 0 <= cur_hdg <= 20)
        elif hdg_idx == 2:
            good_approach = (70 <= cur_hdg <= 110)
        elif hdg_idx == 3:
            good_approach = (210 <= cur_hdg <= 250)
    elif dest_idx == 2:
        if hdg_idx == 1:
            good_approach = (260 <= cur_hdg <= 300)
        elif hdg_idx == 2:
            good_approach = (190 <= cur_hdg <= 230)
    elif dest_idx == 3:
        if hdg_idx == 1:
            good_approach = (190 <= cur_hdg <= 230)
        if hdg_idx == 2:
            good_approach = (10 <= cur_hdg <= 50)
    return good_approach




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

def destination_creator(settings):
    if settings.multi_destination:
        n_destinations = 3
    else:
        n_destinations = 1

    if settings.destination_distribution == "random":
        dest_idx = np.random.randint(n_destinations, size=settings.n_ac) + 1
    elif settings.destination_distribution == "uniform":
        bracket_size = split(settings.n_ac, n_destinations)
        if n_destinations == 3:
            dest_idx = np.concatenate((np.ones(bracket_size[-1], dtype=np.int32)*3, np.ones(bracket_size[-2], dtype=np.int32)*2, np.ones(bracket_size[-3], dtype=np.int32)))
        else:
            dest_idx = np.ones(bracket_size[-1], dtype=np.int32)

        np.random.shuffle(dest_idx)
    else:
        raise ValueError("Destination distribution should be either random or uniform")

    if settings.destination_hdg:
        # Determine approach hdg for destinations if wanted.
        # EHAM: [1: 340-20]
        #       [2: 70-110]
        #       [3: 210-250]
        #
        # EHGR: [1: 260-300]
        #       [2: 190-230]
        #
        # EHDL: [1: 190-230]
        #       [2: 10-40 ]
        dest_hdg_list = np.array([])
        for airport in dest_idx:
            if airport == 1:
                hdg_idx = np.random.randint(3) + 1
                dest_hdg_list = np.append(dest_hdg_list, hdg_idx)
            elif airport == 2:
                hdg_idx = np.random.randint(2) + 1
                dest_hdg_list = np.append(dest_hdg_list, hdg_idx)
            elif airport == 3:
                hdg_idx = np.random.randint(2) + 1
                dest_hdg_list = np.append(dest_hdg_list, hdg_idx)
    else:
        dest_hdg_list = np.ones(settings.n_ac, dtype=np.int32)

    return dest_idx, dest_hdg_list


def split(x, n):
    temp = np.array([])
    if (x % n == 0):
        for _ in range(n):
            temp = np.append(temp, x/n)
    else:
        zp = n - (x % n)
        pp = x // n
        for i in range(n):
            if (i >= zp):
                temp = np.append(temp, pp + 1)
            else:
                temp = np.append(temp, pp)
    return temp.astype(np.int32)
