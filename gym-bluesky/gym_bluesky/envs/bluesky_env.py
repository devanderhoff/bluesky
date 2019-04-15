import numpy as np
import gym
from gym import spaces
import bluesky as bs
from bluesky import tools
from bluesky.network.client import Client
import time


class BlueSkyEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, **kwargs):
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
        self.NodeID = kwargs['NodeID']
        self.n_cpu = kwargs['n_cpu']
        self.cfgfile = ''
        self.scenfile = kwargs['scenfile']
        if self.scenfile is None:
            self.scenfile = './synthetics/super/super3.scn'

        if self.NodeID != 0 and self.n_cpu >= 1:
            # print('Client {0:2d} waiting for node to initialize'.format(self.NodeID))
            time.sleep(5)

        self.myclient = Client()
        self.myclient.connect(event_port=11000, stream_port=11001)
        self.myclient.receive(1000)
        self.myclient.actnode(self.myclient.servers[self.myclient.host_id]['nodes'][self.NodeID])
        self.myclient.event_received.connect(on_event)

        if self.NodeID == 0 and self.n_cpu >= 1:
            str_to_send = 'addnodes ' + str(self.n_cpu - 1)
            # print(str_to_send)
            self.myclient.send_event(b'STACKCMD', str_to_send)
            print('Initializing {0:2d} nodes'.format(self.n_cpu))
            time.sleep(5)

        # Set constants for environment
        self.nm = 1852  # Nautical miles [nm] in meter [m]
        self.ep = 0
        self.los = 5 * 1.05  # [nm] Horizontal separation minimum for resolution
        self.wpt_reached = 5  # [nm]
        self.min_hdg = 0    # [deg]
        self.max_hdg = 360  # [deg]
        self.min_lat = -1  # [lat/lon]
        self.max_lat = 1  # [lat/lon]
        self.min_lon = -1  # [lat/lon]
        self.max_lon = 1  # [lat/lon]

        self.min_dist_plane = self.los * 0.95
        self.max_dist_plane = 125
        self.min_dist_waypoint = self.wpt_reached * 0.95
        self.max_dist_waypoint = 125

        # TODO:State definitions and other state information still has to be formalized.
        self.state = None
        self.state_object = None

        # Define observation bounds and normalize so that all values range between -1 and 1 or 0 and 1,
        # normalization improves neural networks ability to converge
        self.low_obs = np.array([self.min_lat, self.min_lon,
                                 normalizer(self.min_hdg, 'HDGToNorm', self.min_hdg,self.max_hdg),
                                 normalizer(self.min_dist_plane, 'DistToNorm', self.min_dist_plane, self.max_dist_plane),
                                 normalizer(self.min_dist_waypoint, 'DistToNorm', self.min_dist_waypoint, self.max_dist_waypoint)])
        self.high_obs = np.array([self.max_lat, self.max_lon,
                                  normalizer(self.max_hdg, 'HDGToNorm', self.min_hdg, self.max_hdg),
                                  normalizer(self.max_dist_plane, 'DistToNorm', self.min_dist_plane, self.max_dist_plane),
                                  normalizer(self.max_dist_waypoint, 'DistToNorm', self.min_dist_waypoint, self.max_dist_waypoint)])
        self.viewer = None

        # observation is a array of shape (5,) [latitude,longitude,hdg,dist_plane,dist_waypoint]
        self.observation_space = spaces.Box(low=self.low_obs, high=self.high_obs, dtype=np.float32)
        # Action space is normalized heading, shape (1,)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        """
        Reset the environment to initial state
        recdata is a dict that contains requested simulation data from Bluesky. Can be changed in plugin MLCONTROL.
        """
        global recdataflag

        str_to_send = 'IC ' + self.scenfile + '; MLSTEP'
        recdataflag = False
        self.myclient.send_event(b'STACKCMD', str_to_send)

        while not recdataflag:
            self.myclient.receive(1000)

        # Calculate distance to plane and waypoint. This method will change when more research is done into state def.
        dist_plane = tools.geo.kwikdist(recdata['lat'][0], recdata['lon'][0], recdata['lat'][1], recdata['lon'][1])
        dist_wpt = tools.geo.kwikdist(recdata['lat'][0], recdata['lon'][0], recdata['actwptlat'][0], recdata['actwptlon'][0])

        # Set state to initial state.
        self.state = np.array([recdata['lat'][0],recdata['lon'][0],
                               normalizer(recdata['hdg'][0], 'HDGToNorm', self.min_hdg, self.max_hdg),
                               normalizer(dist_plane, 'DistToNorm', self.min_dist_plane, self.max_dist_plane),
                               normalizer(dist_wpt, 'DistToNorm', self.min_dist_waypoint, self.max_dist_waypoint)
                               ])

        self.state_object = np.array([recdata['lat'][1], recdata['lon'][1], recdata['hdg'][1]])
        self.ep = 0

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
        reward = 0
        self.ep = self.ep + 1
        done = False

        # Forward action and let bluesky run a simulation step.
        # Normalize action to be between 0 and 1
        action_tot = normalizer(action[0], 'NormToHDG', self.min_hdg, self.max_hdg)
        self.myclient.send_event(b'STACKCMD', 'HDG SUP0 ' + np.array2string(action_tot) + '; MLSTEP')

        # Wait for server to respond
        while not recdataflag:
            self.myclient.receive(1000)

        # Do some intermediate state update calculations.
        dist_plane = tools.geo.kwikdist(recdata['lat'][0], recdata['lon'][0], recdata['lat'][1], recdata['lon'][1])
        dist_wpt = tools.geo.kwikdist(recdata['lat'][0], recdata['lon'][0], recdata['actwptlat'][0], recdata['actwptlon'][0])

        # Assign rewards if goals states are met or any other arbitary rewards.
        if dist_plane <= self.los:
            reward = reward - 100
            # print('los failure')
            done = True

        if dist_wpt <= self.wpt_reached:
            reward = reward + 100
            # print('wpt reached')
            done = True

        if self.ep >= 1000:
            # print('steps reached')
            done = True

        reward = reward + 3/dist_wpt

        # Update state to state(t + step)
        self.state = np.array([recdata['lat'][0], recdata['lon'][0],
                               normalizer(recdata['hdg'][0], 'HDGToNorm', self.min_hdg, self.max_hdg),
                               normalizer(dist_plane, 'DistToNorm', self.min_dist_plane, self.max_dist_plane),
                               normalizer(dist_wpt, 'DistToNorm', self.min_dist_waypoint, self.max_dist_waypoint)
                               ])

        self.state_object = np.array([recdata['lat'][1], recdata['lon'][1], recdata['hdg'][1]])

        # Check if state is a terminal state, then stop simulation.
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

        screen_width = 600
        screen_height = 600

        scale = screen_width/120

        latplane = self.state[0]
        lonplane = self.state[1]
        latplane2 = self.state_object[0]
        lonplane2 = self.state_object[1]

        length_plane = 1 * scale

        if self.viewer is None:
            #set start position
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            plane = rendering.FilledPolygon([(-length_plane, -length_plane), (0, length_plane), (length_plane, -length_plane)]) # NOSEUP
            x_plane, y_plane = latlon_to_screen(-1, -1, latplane, lonplane, scale)
            plane.add_attr(rendering.Transform(translation=(x_plane, y_plane)))
            self.planetrans = rendering.Transform()
            plane.add_attr(self.planetrans)
            plane.set_color(0, 1, 0)
            self.viewer.add_geom(plane)

            plane2 = rendering.FilledPolygon([(-length_plane, -length_plane), (0, length_plane), (length_plane, -length_plane)]) # NOSEUP
            x_plane2, y_plane2 = latlon_to_screen(-1, -1, latplane2, lonplane2, scale)
            plane2.add_attr(rendering.Transform(translation=(x_plane2, y_plane2)))
            self.planetrans2 = rendering.Transform()
            plane2.add_attr(self.planetrans2)
            plane.set_color(1, 0, 0)
            self.viewer.add_geom(plane2)

            waypoint = rendering.make_circle(3)
            x_waypoint, y_waypoint = latlon_to_screen(-1,-1,bs.traf.actwp.lat[0], bs.traf.actwp.lon[0], scale)
            waypoint.add_attr(rendering.Transform(translation=(x_waypoint, y_waypoint)))
            self.viewer.add_geom(waypoint)

            self.latplane_init = latplane
            self.lonplane_init = lonplane
            self.latplane2_init = latplane2
            self.lonplane2_init = lonplane2

        x_screen_dx, y_screen_dy = dlatlon_to_screen(self.latplane_init, self.lonplane_init, latplane, lonplane, scale)
        self.planetrans.set_translation(x_screen_dx, y_screen_dy)
        # self.planetrans.set_rotation(1)

        x_screen_dx2, y_screen_dy2 = dlatlon_to_screen(self.latplane2_init, self.lonplane2_init, latplane2, lonplane2, scale)
        self.planetrans2.set_translation(x_screen_dx2, y_screen_dy2)
        # self.planetrans2.set_rotation(self.state_object[2]*0.0174532925)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

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


def latlon_to_screen(lat_center, lon_center, latplane, lonplane, scale):
    # Obsolete rendering functions to properly display plane location.
    x_screen = tools.geo.kwikdist(lat_center, lon_center, lat_center, lonplane) * scale
    y_screen = tools.geo.kwikdist(lat_center, lon_center, latplane, lon_center) * scale
    return x_screen, y_screen


def dlatlon_to_screen(lat_center, lon_center, latplane, lonplane, scale):
    # Obsolete rendering functions to properly display plane translation
    qdr, dist = tools.geo.qdrdist(lat_center, lon_center, latplane, lonplane)
    y_screen = dist * np.cos((2*np.pi/360)*qdr) * scale
    x_screen = dist * np.sin((2*np.pi/360)*qdr) * scale
    return x_screen, y_screen
