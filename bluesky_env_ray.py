import numpy as np
from gym import spaces
from bluesky import tools
from ray.rllib.env import MultiAgentEnv
import bluesky as bs

class BlueSkyEnv(MultiAgentEnv):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, env_config):
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
        self.work_id = env_config.worker_index
        self.env_id = env_config.vector_index
        # self.env_per_worker = 2  #set by env_config at some point
        # self.node_id_list = [None] * self.env_per_worker
        self.cfgfile = ''
        self.scenfile = None
        self.connected = False
        if self.scenfile is None:
            self.scenfile = './synthetics/super/super3.scn'

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

        # Define observation bounds and normalize so that all values range between -1 and 1 or 0 and 1,
        # normalization improves neural networks ability to converge
        self.low_obs = np.array([self.min_lat, self.min_lon, self.min_hdg, self.min_dist_waypoint, self.min_dist_plane,
                                 self.min_dist_plane])

        self.high_obs = np.array([self.max_lat, self.max_lon, self.max_hdg, self.max_dist_waypoint, self.max_dist_plane,
                                  self.max_dist_plane])

        # self.low_obs = np.array([self.min_lat, self.min_lon,
        #                          normalizer(self.min_hdg, 'HDGToNorm', self.min_hdg, self.max_hdg),
        #                          normalizer(self.min_dist_waypoint, 'DistToNorm', self.min_dist_waypoint,
        #                                     self.max_dist_waypoint),
        #                          normalizer(self.min_dist_plane, 'DistToNorm', self.min_dist_plane,
        #                                     self.max_dist_plane),
        #                          normalizer(self.min_dist_plane, 'DistToNorm', self.min_dist_plane, self.max_dist_plane)
        #                          ])
        # self.high_obs = np.array([self.max_lat, self.max_lon,
        #                           normalizer(self.max_hdg, 'HDGToNorm', self.min_hdg, self.max_hdg),
        #                           normalizer(self.max_dist_waypoint, 'DistToNorm', self.min_dist_waypoint,
        #                                      self.max_dist_waypoint),
        #                           normalizer(self.max_dist_plane, 'DistToNorm', self.min_dist_plane,
        #                                      self.max_dist_plane),
        #                           normalizer(self.max_dist_plane, 'DistToNorm', self.min_dist_plane,
        #                                      self.max_dist_plane)
        #                           ])
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
        if not self.connected:
            bs.init()
            bs.net.connect()
            self.connected = True
        str_to_send = 'IC ' + self.scenfile
        bs.stack.stack(str_to_send)

        simstep()
        simstep()
        bs.sim.fastforward()
        recdata = dict(
            lat=bs.traf.lat,
            lon=bs.traf.lon,
            hdg=bs.traf.hdg,
            actwplat=bs.traf.actwp.lat,
            actwplon=bs.traf.actwp.lon,
            id=bs.traf.id
        )
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

