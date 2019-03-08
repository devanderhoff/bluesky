#!/usr/bin/env python
""" Main BlueSky start script """
from __future__ import print_function
import sys
import traceback
import bluesky as bs
from bluesky.ui import qtgl
from bluesky import stack, settings, navdb, traf, sim, scr, tools
import numpy as np
from bluesky.tools import timer
import gym
import pygame as pg
from bluesky.ui.pygame import splash

# Create custom system-wide exception handler. For now it replicates python's
# default traceback message. This was added to counter a new PyQt5.5 feature
# where unhandled exceptions would result in a qFatal with a very uninformative
# message.
def exception_handler(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit()


sys.excepthook = exception_handler



def main():

    mode = 'sim-detached'
    # mode = 'server - headless'


    discovery = False

    # Check if alternate config file is passed or a default scenfile
    cfgfile = ''
    scnfile = '/synthetics/super/super2.scn'


    bs.init(mode, discovery=discovery, cfgfile=cfgfile, scnfile=scnfile)
    bs.sim.step()
    # print_state()
    # print(np.array([bs.traf.lat[0], bs.traf.lon[0], bs.traf.hdg[0]], bs.traf.asas.dist[0]))
    for i in range(10000):

        bs.sim.step()
        print_location()
        # dist_plane = tools.geo.kwikdist(bs.traf.lat[0],bs.traf.lon[0],bs.traf.lat[1],bs.traf.lon[1])
        # dist_wpt = tools.geo.kwikdist(bs.traf.lat[0],bs.traf.lon[0],bs.traf.actwp.lat[0],bs.traf.actwp.lon[0])
        # state = np.array([bs.traf.lat[0], bs.traf.lon[0], bs.traf.hdg[0],dist_plane,dist_wpt])
        # bs.stack.stack(bs.traf.id[0] + ' HDG ' + '120')
        # print(state)
        # print(i)

def print_state():
    dist_plane = tools.geo.kwikdist(bs.traf.lat[0],bs.traf.lon[0],bs.traf.lat[1],bs.traf.lon[1])
    dist_wpt = tools.geo.kwikdist(bs.traf.lat[0],bs.traf.lon[0],bs.traf.actwp.lat[0],bs.traf.actwp.lon[0])
    state = np.array([bs.traf.lat[0], bs.traf.lon[0], bs.traf.hdg[0],dist_plane,dist_wpt])
    print(state)
    return

def print_location():
    print(bs.traf.lat)
    print(bs.traf.lon)
    print(tools.geo.kwikdist(bs.traf.lat[0],bs.traf.lon[0],bs.traf.actwp.lat[0],bs.traf.actwp.lon[0]))
if __name__ == "__main__":
    main()