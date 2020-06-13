""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, settings, navdb, traf, sim, scr, tools
import numpy as np
from bluesky.tools import geo
import csv

# import ExternalEnv

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'EXAMPLE',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 2.5,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.
        'update':          update,

        # The preupdate function is called before traffic is updated. Use this
        # function to provide settings that need to be used by traffic in the current
        # timestep. Examples are ASAS, which can give autopilot commands to resolve
        # a conflict.
        'preupdate':       preupdate,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        'reset':         reset
        }

    stackfunctions = {
        # The command name for your function
        'MYFUN': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'MYFUN ON/OFF',

            # A list of the argument types your function accepts. For a description of this, see ...
            '[onoff]',

            # The name of your function in this plugin
            myfun,

            # a longer help text of your function.
            'Print something to the bluesky console based on the flag passed to MYFUN.']
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin
first_time = True
done_count = 0
def update():
    global first_time, done, delete_dict, done_count
    if first_time:
        done = dict.fromkeys(traf.id)
        delete_dict = dict.fromkeys(traf.id, False)
        first_time = False

    for id in traf.id:
        if id not in done:
            done[id] = False
        if id not in delete_dict:
            delete_dict[id] = False
    dest_lat_lon = [(52.6, 4.73), (52.3, 4.36), (52.33, 5.19), (51.52, 5.33), (51.8, 5.06), (51.82, 5.75), (52.30, 6.0)]
    # dest_lat_lon[-dest]

    if done.keys():
        for agent_id in done.keys():
            # Initialize reward to 0.
            done[agent_id] = False
            # First check if goal area is reached
            idx = traf.id2idx(agent_id)
            dest = traf.dest_temp[idx]
            dest_lat, dest_lon = dest_lat_lon[-dest]
            dist = geo.kwikdist(traf.lat[idx], traf.lon[idx], dest_lat, dest_lon)
            if dist <= 5:
                done[agent_id] = True
                done_count +=1

        for agent_id in done.keys():
            if agent_id in delete_dict:
                if done[agent_id] and not delete_dict[agent_id]:
                    traf.delete(traf.id2idx(agent_id))
                    print('Deleted ', agent_id)
                    delete_dict[agent_id] = True
    save_metrics()
    if done_count >= 125:
        sim.reset()

    return

def preupdate():
    return
def reset():
    global first_time_csv, crash_count
    first_time_csv = True
    crash_count = 0
    return

### Other functions of your plugin
def myfun(flag=True):
    return True, 'My plugin received an o%s flag.' % ('n' if flag else 'ff')

def retrieve_state():

    pass

def save_metrics():
    global first_time_csv
    f = open('save.csv', 'a')
    writer = csv.writer(f)
    if first_time_csv:
        # writer = csv.writer(f)
        writer.writerow(['time', 'crashes', 'landed']) #crash_count, done_count-crash_count])
        first_time_csv = False

    writer.writerow([sim.simt, crash_count, done_count-crash_count])
    return
