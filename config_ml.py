## Config file for the ML part of this project. Using this instead of settings.cfg because of ease of use.

#=============================================================================
#=  RLlib plugin settings
#=  Pluging mlcontrolc has to be enabled for these settings to have effect
#=============================================================================
import pickle
# #Number of a/c neighbours to take into account n_neighbours is always maximum: n_ac - 1.
# n_neighbours = 3
# n_ac = 5
#
# #Update of policy enabled/disabled
# training_enabled = False
#
# #Airspeed at creation of aircraft
# acspd = 250
#
# #Number of a/c neighbours to take into account n_neighbours is always maximum: n_ac - 1.
# n_neighbours = 3
# print(n_neighbours)
# assert n_ac - n_neighbours >= 1, "Nr ac_neighbours should always at maximum be n_ac - 1"
#
# print(n_neighbours)
# assert n_ac - n_neighbours >= 1, "Nr ac_neighbours should always at maximum be n_ac - 1"
#
# #min lat
# min_lat = 51
# max_lat = 54
# min_lon = 2
# max_lon = 8

class Config:
    def __init__(self):
        # Configs for config geh
        # External trainer configs, such as names, ports and other misc flags
        self.save_file_name = None
        self.max_concurrent = None  # Concurrent episodes on REST server
        self.checkpoint_file_name = None  # Save name for trainer checkpoints.
        self.multiagent = None  # Flag for multiagent approach
        self.training_enabled = None  # Update of policy enabled/disabled
        self.server_port = None  # Listerning port for REST server (policy)
        self.evaluate = None    # If True, do not reset environment. Used for execution of scenfiles.

        # Environment settings
        self.n_ac = None                    # Nr of aircraft created
        self.n_neighbours = None            # Nr of aircraft taken into account for local observations
        self.min_lat = None                 # Minimum latitude in observation space
        self.min_lat_gen = None             # Minimum lat for a/c generation
        self.max_lat = None                 # Maximum latitude in observation space
        self.max_lat_gen = None             # Maximum lat for a/c generation
        self.min_lon = None                 # Minimum longitude in observation space
        self.min_lon_gen = None             # Minimum lon for a/c generation
        self.max_lon = None                 # Maximum longitude in observation space
        self.max_lon_gen = None             # Maximum lon for a/c generation
        self.min_dist = None                # Minimum distance between A/C
        self.max_dist = None                # Maximum distance between A/C (obs space)
        self.wpt_almost_reached = None             # When a A/C is considered to have reached its goal (nm)
        self.los = None                     # When an A/C is considered to have a loss of seperation (nm)

        # Destination settings
        self.spawn_separation = None
        self.lat_eham = None                # Latitude goal     (to be extended)
        self.lon_eham = None                # Longitude goal    (to be extended)
        self.destination_distribution = None # If the destination should be randomly distributed, or an uniform distribution across all airports should be done. Either "random" or "uniform"
        self.destination_hdg = None              # Use hdg limitations for each destination.
        self.multi_destination = None       # Using multiple destinations or not

       # Trainer settings
        self.max_timesteps = None           # Max timesteps per episode
        self.gamma = None                   # Discount rate



    def set_val(self, in_dct):
        # Sets attribute value
        for key in in_dct.keys():
            setattr(self, key, in_dct[key])
        self.check()
        self.check_constraints()
        self.save_conf()


    def check_constraints(self):
        print('Check settings constraints')
        assert self.n_ac - self.n_neighbours >= 0, "Nr ac_neighbours should always at maximum be n_ac - 1"
        assert isinstance(self.save_file_name, str), 'save file naming should be string'
        assert (self.destination_distribution == "random" or self.destination_distribution == "uniform"), \
            "Wrong destination distribution given"
        assert (self.destination_hdg is not False or self.destination_hdg is not True), 'settings has to be set.'
        assert (self.multi_destination is not True or self.multi_destination is not False), 'settings has to be set'
        # assert (self.evaluate is True and self.training_enabled is False), 'If evaluating, disable training'

    def check(self):
        prop = vars(self)
        for key in prop.keys():
            if prop[key] is None:
                raise ValueError('Not all required config variables are set, check the input dict')

    def save_conf(self):
        sv = open(self.save_file_name, 'wb')
        pickle.dump(self, sv)
        sv.close()

    def load_conf(self, save_file_name):
        save_file_name = '/home/dennis/PycharmProjects/bluesky/' + save_file_name
        return pickle.load(open(save_file_name, 'rb'))
