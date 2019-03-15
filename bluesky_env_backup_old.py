import numpy as np

import gym
from gym import spaces
import bluesky as bs
from bluesky import tools
import sys
from bluesky.network.client import Client


class BlueSkyEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,**kwargs):
        def on_event(eventname, eventdata, sender_id):
            print(eventdata)
            print(sender_id)

        def on_stream(streamname, data, sender_id):
            print('Stream data received:', streamname)
            self.received_data = True
            self.data = data

        ## Enable client server interface
        self.NodeID = kwargs['NodeID']
        self.received_data = False
        myclient = Client()
        myclient.connect(event_port=11000, stream_port=11001)
        myclient.receive()
        myclient.actnode(myclient.servers[myclient.host_id]['nodes'][self.NodeID])
        myclient.subscribe(b'ACDATA')


        myclient.stream_received.connect(on_stream)
        myclient.event_received.connect(on_event)

        while not self.received_data:
            myclient.receive()






        self.nm = 1852
        self.ep = 0
        self.los = 5 * 1.05  # [nm] Horizontal separation minimum for resolution
        self.wpt_reached = 5
        # observation is a array of 4 collums [latitude,longitude,hdg,dist_plane,dist_waypoint] Real values, so not yet normalized
        self.min_hdg = 0 #0
        self.max_hdg = 360 #360
        self.min_lat = -1
        self.max_lat = 1
        self.min_lon = -1
        self.max_lon = 1
        # max values based upon size of latlon box
        self.min_dist_plane = self.los * 0.95
        self.max_dist_plane = 125
        self.min_dist_waypoint = self.wpt_reached * 0.95
        self.max_dist_waypoint = 125

        # define observation bounds
        self.low_obs = np.array([self.min_lat, self.min_lon,
                                 normalizer(self.min_hdg,'HDGToNorm',self.min_hdg,self.max_hdg),
                                 normalizer(self.min_dist_plane,'DistToNorm', self.min_dist_plane, self.max_dist_plane),
                                 normalizer(self.min_dist_waypoint, 'DistToNorm', self.min_dist_waypoint, self.max_dist_waypoint)])
        self.high_obs = np.array([self.max_lat, self.max_lon,
                                  normalizer(self.max_hdg, 'HDGToNorm', self.min_hdg, self.max_hdg),
                                  normalizer(self.max_dist_plane, 'DistToNorm', self.min_dist_plane, self.max_dist_plane),
                                  normalizer(self.max_dist_waypoint, 'DistToNorm', self.min_dist_waypoint, self.max_dist_waypoint)])
        self.viewer = None



        # Initialize bluesky
        self.mode = 'sim-detached'
        self.discovery = ('--discoverable' in sys.argv or self.mode[-8:] == 'headless')
        # Check if alternate config file is passed or a default scenfile
        self.cfgfile = ''
        self.scnfile = '/synthetics/super/super3.scn'
        bs.init(self.mode, discovery=self.discovery, cfgfile=self.cfgfile, scnfile=self.scnfile)
        bs.sim.fastforward()
        self.observation_space = spaces.Box(low=self.low_obs, high=self.high_obs, dtype=np.float32)
        # self.action_space = spaces.Discrete(360)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # self.reset()



    def reset(self):
        # reset and reinitizalie sim
        bs.sim.reset()
        bs.init(self.mode, discovery=self.discovery, cfgfile=self.cfgfile, scnfile=self.scnfile)
        bs.sim.step()
        bs.sim.fastforward()
        # calculate state values
        dist_plane = tools.geo.kwikdist(bs.traf.lat[0], bs.traf.lon[0], bs.traf.lat[1], bs.traf.lon[1])
        dist_wpt = tools.geo.kwikdist(bs.traf.lat[0], bs.traf.lon[0], bs.traf.actwp.lat[0], bs.traf.actwp.lon[0])
        # create initial observation is a array of 4 collums [latitude,longitude,hdg,dist_plane,dist_waypoint]
        self.state = np.array([bs.traf.lat[0],bs.traf.lon[0],
                               normalizer(bs.traf.hdg[0], 'HDGToNorm', self.min_hdg, self.max_hdg),
                               normalizer(dist_plane, 'DistToNorm', self.min_dist_plane, self.max_dist_plane),
                               normalizer(dist_wpt, 'DistToNorm', self.min_dist_waypoint, self.max_dist_waypoint)
                               ])

        # self.state = np.array([bs.traf.lat[0], bs.traf.lon[0], bs.traf.hdg[0], dist_plane, dist_wpt])
        self.state_object = np.array([bs.traf.lat[1], bs.traf.lon[1], bs.traf.hdg[1]])
        self.ep = 0
        return self.state

    def step(self, action):
        reward = 0
        self.ep = self.ep + 1
        done = False
        #do one step and perform action
        # action_str = np.array2string(action[0])
        # action = np.round(action)
        #relative heading
        action_tot = normalizer(action[0],'NormToHDG', self.min_hdg, self.max_hdg)
        # if self.ep/100 in {1,2,3,4,5,6,7,8,9}:
        #     print(action[0])
        #     print(action_tot)

        bs.stack.stack(bs.traf.id[0] + ' HDG ' + np.array2string(action_tot))
        bs.sim.step()
        dist_plane = tools.geo.kwikdist(bs.traf.lat[0], bs.traf.lon[0], bs.traf.lat[1], bs.traf.lon[1])
        dist_wpt = tools.geo.kwikdist(bs.traf.lat[0], bs.traf.lon[0], bs.traf.actwp.lat[0], bs.traf.actwp.lon[0])
        if dist_plane <= self.los:
            reward = reward - 100
            done = True

        if dist_wpt <= self.wpt_reached:
            reward = reward + 100
            done = True
        self.state = np.array([bs.traf.lat[0], bs.traf.lon[0],
                               normalizer(bs.traf.hdg[0], 'HDGToNorm', self.min_hdg, self.max_hdg),
                               normalizer(dist_plane, 'DistToNorm', self.min_dist_plane, self.max_dist_plane),
                               normalizer(dist_wpt, 'DistToNorm', self.min_dist_waypoint, self.max_dist_waypoint)
                               ])
        # self.state = np.array([bs.traf.lat[0], bs.traf.lon[0], bs.traf.hdg[0], dist_plane, dist_wpt])
        self.state_object = np.array([bs.traf.lat[1], bs.traf.lon[1], bs.traf.hdg[1]])
        # print(dist_plane)
        # print(self.state)
        # calculate reward
        # print(action_str)
        reward = reward - dist_wpt
        # reward = reward + -dist_plane/5

        # check LOS and wpt reached


        if self.ep >= 500:
            done = True

        # print(reward)
        return self.state, reward, done, {}

    def render(self, mode='human'):

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




def normalizer(value,type,min,max):
    #Scales input values so that the NN performance is improved
    #Currently scales to fixed determined values depending on the needed scale.
    #List of scales:
    #HDG : from 1/360 to -1,1
    #Distances scales from min/max_dist to 0,1
    #latlon from lat/lon to -1,1
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
    x_screen = tools.geo.kwikdist(lat_center, lon_center, lat_center, lonplane) * scale
    y_screen = tools.geo.kwikdist(lat_center, lon_center, latplane, lon_center) * scale
    return x_screen, y_screen

def dlatlon_to_screen(lat_center, lon_center, latplane, lonplane, scale):
    qdr, dist = tools.geo.qdrdist(lat_center, lon_center, latplane, lonplane)
    y_screen = dist * np.cos((2*np.pi/360)*qdr) * scale
    x_screen = dist * np.sin((2*np.pi/360)*qdr) * scale
    return x_screen, y_screen
