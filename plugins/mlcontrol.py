""" External control plugin for Machine Learning applications. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, sim, traf  #, settings, navdb, traf, sim, scr, tools
from ray.rllib.env import ExternalEnv
import numpy as np
from gym import spaces
from bluesky import tools
# from ray.rllib.env import MultiAgentEnv
import bluesky as bs

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code
    global env
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'MLCONTROL',

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
    nm = 1852  # Nautical miles [nm] in meter [m]
    ep = 0
    los = 1.05  # [nm] Horizontal separation minimum for resolution
    wpt_reached = 5  # [nm]
    min_hdg = 1  # [deg]
    max_hdg = 360  # [deg]
    min_lat = 51  # [lat/lon]
    max_lat = 54  # [lat/lon]
    min_lon = 2  # [lat/lon]
    max_lon = 8  # [lat/lon]

    min_dist_plane = los * 0.95
    max_dist_plane = 80000
    min_dist_waypoint = wpt_reached * 0.95
    max_dist_waypoint = 80000

    low_obs = np.array(
        [min_lat - 10, min_lon - 10, min_hdg, min_dist_waypoint, min_dist_plane,
         min_dist_plane])

    high_obs = np.array(
        [max_lat + 10, max_lon + 10, max_hdg, max_dist_waypoint, max_dist_plane,
         max_dist_plane])

    # observation is a array of shape (5,) [latitude,longitude,hdg,dist_plane,dist_waypoint]
    observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
    # Action space is normalized heading, shape (1,)
    action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    env = BlueSkyEnv(action_space, observation_space, 100)
    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

def update():
    data = dict(
        lat=traf.lat,
        lon=traf.lon,
        hdg=traf.hdg,
        actwplat=traf.actwp.lat,
        actwplon=traf.actwp.lon,
        id=traf.id,
    )
    env.run(data)
    sim.send_event(b'MLSTATEREPLY', data, myclientrte)
    sim.hold()


def preupdate():
    pass

def reset():
    pass

def mlstep():
    global myclientrte
    myclientrte = stack.routetosender()
    sim.fastforward
    sim.op()



class BlueSkyEnv(ExternalEnv):
    def __init__(self, action_space, observation_space, max_concurrent):
        """
                # Initialize client server interface
                # NodeID: is required for each individual environment to connect to its respective simulation node, when
                # running a single simulation this argument is not required.
                # n_cpu: total amount of nodes that have to be initialized.
                # NOTE: N_cpu and len(NodeID) have to be the same
                # scenfile: The specific scenario file this environment is going to run.
                # TODO:
                    -Use bs.settings for ports and plugin check.
                    -Starting of server integration (Now the server has to separately be started by using bluesky.py --headless
                """
        # Inherit __init__
        ExternalEnv.__init__(self, action_space, observation_space, max_concurrent)

        # Fill in inherited properties, obs and action.
        # Define observation bounds and normalize so that all values range between -1 and 1 or 0 and 1,
        # normalization improves neural networks ability to converge
        # Set constants for environment
        self.nm = 1852  # Nautical miles [nm] in meter [m]
        self.ep = 0
        self.los = 1.05  # [nm] Horizontal separation minimum for resolution
        self.wpt_reached = 5  # [nm]
        self.min_hdg = 1    # [deg]
        self.max_hdg = 360  # [deg]
        self.min_lat = 51  # [lat/lon]
        self.max_lat = 54  # [lat/lon]
        self.min_lon = 2  # [lat/lon]
        self.max_lon = 8  # [lat/lon]

        self.min_dist_plane = self.los * 0.95
        self.max_dist_plane = 80000
        self.min_dist_waypoint = self.wpt_reached * 0.95
        self.max_dist_waypoint = 80000

        self.low_obs = np.array(
            [self.min_lat - 10, self.min_lon - 10, self.min_hdg, self.min_dist_waypoint, self.min_dist_plane,
             self.min_dist_plane])

        self.high_obs = np.array(
            [self.max_lat + 10, self.max_lon + 10, self.max_hdg, self.max_dist_waypoint, self.max_dist_plane,
             self.max_dist_plane])

        # observation is a array of shape (5,) [latitude,longitude,hdg,dist_plane,dist_waypoint]
        self.observation_space = spaces.Box(low=self.low_obs, high=self.high_obs, dtype=np.float32)
        # Action space is normalized heading, shape (1,)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Env_config
        self.work_id = None
        self.env_id = None
        self.env_per_worker = 2  #set by env_config at some point
        # self.node_id_list = [None] * self.env_per_worker

        # BlueSky init config
        self.cfgfile = ''
        self.scenfile = None
        self.connected = False
        if self.scenfile is None:
            self.scenfile = './synthetics/init_scen_ray.scn'

        # Set constants for environment
        self.nm = 1852  # Nautical miles [nm] in meter [m]
        self.ep = 0
        self.los = 1.05  # [nm] Horizontal separation minimum for resolution
        self.wpt_reached = 5  # [nm]
        self.min_hdg = 1    # [deg]
        self.max_hdg = 360  # [deg]
        self.n_ac = 3
        self.state = None

    def run(self, obs):
        """Override this to implement the run loop.
        Your loop should continuously:
            1. Call self.start_episode(episode_id)
            2. Call self.get_action(episode_id, obs)
                    -or-
                    self.log_action(episode_id, obs, action)
            3. Call self.log_returns(episode_id, reward)
            4. Call self.end_episode(episode_id, obs)
            5. Wait if nothing to do.
        Multiple episodes may be started at the same time.
        """
        episode_id = self.start_episode()
        self.get_action(episode_id=episode_id, observation=obs)
        reward = 1
        self.log_returns(episode_id, reward, info=None)
        self.end_episode(episode_id, obs)


    def reset(self):
        """
        Reset the environment to initial state
        recdata is a dict that contains requested simulation data from Bluesky. Can be changed in plugin MLCONTROL.
        """
        # Connect to the BlueSky server and pan to area.
        if not self.connected:
            #if self.env_id == 1:
            bs.init(mode="sim-detached")
            #else:
                #bs.init()
                #bs.net.connect()

            self.connected = True
            str_to_send = 'IC ' + self.scenfile
            bs.stack.stack(str_to_send)
            simstep()
            simstep()

        bs.sim.reset()
        # bs.sim.fastforward()

        ## Create aircraft
        # Randomize location withing netherlands
        aclat = np.random.randn(self.n_ac) * (self.max_lat - self.min_lat) + self.min_lat
        aclon = np.random.randn(self.n_ac) * (self.max_lon - self.min_lon) + self.min_lon
        achdg = np.random.randint(1, 360, self.n_ac)
        acspd = np.ones(self.n_ac) * 250
        bs.traf.create(n=self.n_ac, aclat=aclat, aclon=aclon, achdg=achdg, acspd=acspd)

        recdata = dict(
            lat=bs.traf.lat,
            lon=bs.traf.lon,
            hdg=bs.traf.hdg,
            actwplat=bs.traf.actwp.lat,
            actwplon=bs.traf.actwp.lon,
            id=bs.traf.id
        )

        simstep()
        # set properties based on loaded scenfile
        self.nr_agents = len(recdata['id'])
        self.id_agents = recdata['id']

        self.init_agent = dict.fromkeys(self.id_agents)
        for id in self.id_agents:
            self.init_agent[id] = False

        self.state = BlueSkyEnv.calc_state(self, recdata)
        self.ep = 0

        return self.state

    def step(self, action):
        """
        This function interacts with the Bluesky simulation node/server, and gives a action command to the environment
        and returns state(t + step)
        Reward accumulation is done outside this step function, so this function returns the reward gained during
        this simulation step.
        """
        # reward = 0
        self.ep = self.ep + 1

        # Loop over every agent
        # bs.sim.fastforward()
        # bs.stack.stack('HDG ' + 'SUP0' + ' ' + str(action))
        for id, value in action.items():
            action_tot = normalizer(value[0], 'NormToHDG', self.min_hdg, self.max_hdg)
            bs.stack.stack('HDG ' + id + ' ' + np.array2string(action_tot))
        # print(bs.stack.stack)
        simstep()

        recdata = dict(
            lat=bs.traf.lat,
            lon=bs.traf.lon,
            hdg=bs.traf.hdg,
            actwplat=bs.traf.actwp.lat,
            actwplon=bs.traf.actwp.lon,
            id=bs.traf.id
        )
        print(bs.traf.id)
        print(bs.traf.hdg)

        # Forward action and let bluesky run a simulation step.
        # Normalize action to be between 0 and 1

        self.nr_agents = len(recdata['id'])
        self.id_agents = recdata['id']

        #Create state dict
        self.state = BlueSkyEnv.calc_state(self, recdata)

        reward, done = BlueSkyEnv.calc_reward(self)

        # Check if state is a terminal state, then stop simulation.
        # if self.ep >= 500:
        #     done["__all__"] = True

        if done["__all__"]:
            for key in done.keys():
                done[key] = True
            bs.stack.stack('HOLD')

        for id, value in self.init_agent.items():
            if value:
                bs.stack.stack('DEL ' + id)

        return self.state, reward, done, {}

    def calc_state(self, recdata):
        state = dict.fromkeys(self.id_agents)

        for i, id in enumerate(self.id_agents):
            dist_wpt = tools.geo.kwikdist(recdata['lat'][i], recdata['lon'][i], recdata['actwplat'][i],
                                          recdata['actwplon'][i])
            state[id] = np.array([recdata['lat'][i], recdata['lon'][i],recdata['hdg'][i],dist_wpt])

            for j in range(len(self.id_agents)):
                if i is not j:
                    temp = tools.geo.kwikdist(recdata['lat'][i], recdata['lon'][i], recdata['lat'][j], recdata['lon'][j])
                    state[id] = np.append(state[id], temp)

            while len(state[id]) < 6:
                state[id] = np.append(state[id],99)
        return state

    def calc_reward(self):
        #create initialized dicts
        failure = False
        reward = dict.fromkeys(self.id_agents)
        done = dict.fromkeys(self.id_agents)

        for id in self.id_agents:
            reward[id] = 0
            done[id] = False
            done['__all__'] = False

        #Determine rewards and if goals are met
        for id in self.id_agents:
            for i in range(4, self.nr_agents+3):
                if self.state[id][i] <= self.los:
                    reward[id] = reward[id] - 100
                    done["__all__"] = True
                    failure = True

            if self.state[id][3] <= self.wpt_reached:
                reward[id] = reward[id] + 100
                self.init_agent[id] = True
                done[id] = True

            reward[id] = reward[id] + (3 / normalizer(self.state[id][3], 'NormToDist', self.min_dist_waypoint, self.max_dist_waypoint))

        if not failure:
            done["__all__"] = all(value for value in self.init_agent.values()) #need fixing
        return reward, done

def simstep():
    bs.net.step()
    bs.sim.step()

def normalizer(value, type, min, max):
    """
    Scales input values so that the NN performance is improved
    Currently scales to fixed determined values depending on the needed scale.
    List of scales:
    HDG : from 1/360 to -1,1
    Distances scales from min/max_dist to 0,1
    latlon from lat/lon to -1,1
    """
    if type == 'HDGToNorm':
        output = (2/360)*(value-360) + 1
    if type == 'DistToNorm':
        output = (1/(max-min))*(value-max) + 1
    if type == 'NormToHDG':
        output = (360/2)*(value-1) + 360
    if type == 'NormToDist':
        output = (max-min)*(value-1) + max
    return output



