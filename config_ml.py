## Config file for the ML part of this project. Using this instead of settings.cfg because of ease of use.

#=============================================================================
#=  RLlib plugin settings
#=  Pluging mlcontrolc has to be enabled for these settings to have effect
#=============================================================================

#Number of a/c neighbours to take into account n_neighbours is always maximum: n_ac - 1.
n_neighbours = 3
n_ac = 5

#Update of policy enabled/disabled
training_enabled = False

#Airspeed at creation of aircraft
acspd = 250

#Number of a/c neighbours to take into account n_neighbours is always maximum: n_ac - 1.
n_neighbours = 3
print(n_neighbours)
assert n_ac - n_neighbours >= 1, "Nr ac_neighbours should always at maximum be n_ac - 1"

print(n_neighbours)
assert n_ac - n_neighbours >= 1, "Nr ac_neighbours should always at maximum be n_ac - 1"

#min lat
min_lat = 51
max_lat = 54
min_lon = 2
max_lon = 8

class Config:
    def __init__(self):
        self.save_file_name = None
        self.n_ac = None                # Nr of aircraft created
        self.training_enabled = None    # Update of policy enabled/disabled
        self.n_neighbours = None        # Nr of aircraft taken into account for local observations
        self.min_lat = None             # Minimum latitude in observation space
        self.max_lat = None             # Maximum latitude in observation space
        self.min_lon = None             # Minimum longitude in observation space
        self.max_lon = None             # Maximum longitude in observation space


    def set_val(self, in_dct):
        # Sets attribute value
        for key in in_dct.keys():
            setattr(self, key, in_dct[key])
        self.check()
        self.check_constraints()


    def check_constraints(self):
        assert self.n_ac - self.n_neighbours >= 1, "Nr ac_neighbours should always at maximum be n_ac - 1"

    def check(self):
        prop = vars(self)
        for key in prop.keys():
            if prop[key] is None:
                raise ValueError('Not all required config variables are set, check the input dict')

    def save_conf(self):
        pass
