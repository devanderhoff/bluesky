import numpy as np
import gym
from gym import spaces
import bluesky as bs
from bluesky import tools
from bluesky.network.client import Client
import time

class BlueSkyEnv2(gym.Env):

    def __init__(self):
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
        global recdataflag, recdata
        recdataflag = False
        recdata = None

        self.connected = False

        self.scenfile = './synthetics/super/super2.scn'

        self.min_hdg = 0    # [deg]
        self.max_hdg = 360  # [deg]
        self.min_lat = -1  # [lat/lon]
        self.max_lat = 1  # [lat/lon]
        self.min_lon = -1  # [lat/lon]
        self.max_lon = 1  # [lat/lon]

        self.min_dist_waypoint = 5
        self.max_dist_waypoint = 125

        # Define observation bounds and normalize so that all values range between -1 and 1 or 0 and 1,
        # normalization improves neural networks ability to converge
        self.low_obs = np.array([self.min_lat, self.min_lon, self.min_hdg, self.min_dist_waypoint])

        self.high_obs = np.array([self.max_lat, self.max_lon, self.max_hdg, self.max_dist_waypoint])

        # observation is a array of shape (4,) [latitude,longitude,hdg,dist_plane,dist_waypoint]
        self.observation_space = spaces.Box(low=self.low_obs, high=self.high_obs, dtype=np.float32)
        # Action space is normalized heading, shape (1,)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)


    def reset(self):
        """
        Reset the environment to initial state
        recdata is a dict that contains requested simulation data from Bluesky. Can be changed in plugin MLCONTROL.
        """
        global recdataflag
        self.ep = 0
        if not self.connected:
            self.NodeID = 0
            self.myclient = Client()
            self.myclient.connect(event_port=52000, stream_port=52001)
            self.myclient.receive(1000)
            self.myclient.event_received.connect(on_event)
            self.myclient.actnode(self.myclient.servers[self.myclient.host_id]['nodes'][self.NodeID])
            self.connected = True

        #Initialize env.
        str_to_send = 'IC ' + self.scenfile + '; MLSTEP'
        recdataflag = False
        self.myclient.send_event(b'STACKCMD', str_to_send)

        while not recdataflag:
            self.myclient.receive(1000)

        dist_wpt = tools.geo.kwikdist(recdata['lat'][0], recdata['lon'][0], recdata['actwplat'][0], recdata['actwplon'][0])
        self.state = np.array([recdata['lat'][0],recdata['lon'][0],
                               normalizer(recdata['hdg'][0], 'HDGToNorm', self.min_hdg, self.max_hdg),
                               normalizer(dist_wpt, 'DistToNorm', self.min_dist_waypoint, self.max_dist_waypoint)
                               ])
        # Reset data flag when data has been processed.
        recdataflag = False
        return self.state

    def step(self, action):
        """
        This function interacts with the Bluesky simulation node/server, and gives a action command to the environment
        and returns state(t + step)
        Reward accumulation is done outside this step function, so this function returns the reward gained during
        this simulation step.
        """
        # Initialize step parameters
        global recdataflag
        done = False
        reward = 0
        self.ep = self.ep + 1

        # Loop over every agent
        action_tot = normalizer(action[0], 'NormToHDG', self.min_hdg, self.max_hdg)
        self.myclient.send_event(b'STACKCMD', 'SUP0 HDG ' + np.array2string(action_tot) + '; MLSTEP')

        # Wait for server to respond
        while not recdataflag:
            self.myclient.receive(1000)

        # Do some intermediate state update calculations.
        dist_wpt = tools.geo.kwikdist(recdata['lat'][0], recdata['lon'][0], recdata['actwplat'][0], recdata['actwplon'][0])

        # Assign rewards if goals states are met or any other arbitary rewards.

        if dist_wpt <= self.min_dist_waypoint:
            reward = reward + 10
            done = True

        if self.ep >= 500:
            done = True

        #Penalty for every timestep
        reward = reward - 0.1

        # Update state to state(t + step)
        self.state = np.array([recdata['lat'][0], recdata['lon'][0],
                               normalizer(recdata['hdg'][0], 'HDGToNorm', self.min_hdg, self.max_hdg),
                               normalizer(dist_wpt, 'DistToNorm', self.min_dist_waypoint, self.max_dist_waypoint)
                               ])

        if done is True:
            self.myclient.send_event(b'STACKCMD', 'HOLD')

        # Reset data flag when data has been processed.
        recdataflag = False

        return self.state, reward, done, {}

    def render(self, mode='human'):
        """
        Obsolete rendering platform, for debugging purposes.
        To render simulation, start bluesky --client and connect to running server.
        """
        return

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def on_event(eventname, eventdata, sender_id):
    # Function that communicates with the bluesky server data stream.
    global recdataflag, recdata
    if eventname == b'MLSTATEREPLY':
        recdataflag = True
        recdata = eventdata


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

if __name__ == '__main__':
     env = BlueSkyEnv2()
     obs = env.reset()
     print('Initial state: [lat, lon, hdg, dist_wpt]')
     print(obs)
     for i in range(10000):

        action = MLtrainer(state,reward)
        print('Send HDG change of ', normalizer(action[0], 'NormToHDG', 0, 360))

        state, reward = env.step(action)


        print('State transition after HDG change: [lat, lon, hdg, dist_wpt] ')
        print(obs)
        print('Accumulated reward: ', reward)
