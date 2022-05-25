#! /usr/bin/env python 

import numpy as np
from math import ceil

class HATS():
    def __init__(self, sensor_size, tau, tw, cell_size, surface_size):
        self._w, self._h, self._npols = sensor_size
        self._tau, self._tw = np.array(tau, dtype=np.float32), np.array(tw, dtype=np.float32)
        self._cdim, self._sdim = cell_size, surface_size
        # Radius of the time surface.
        self._rho = surface_size//2
        # Number of cell along the horizontal and vertical axes, respectively.
        self._hgrid, self._wgrid = ceil(self._h/self._cdim), ceil(self._w/self._cdim)
        # Total number of cells.
        self._ncells = self._hgrid*self._wgrid
        # Local memories to which the events are mapped.
        self._gen_locmems = lambda: [[[] for p in range(self._npols)] for c in range(self._ncells)]
        # Mapping matrix used to obtain the index of the cell to which an event belongs, given its coordinates.
        self._px_to_cell = np.array([[(y//self._cdim)*self._wgrid + x//self._cdim for x in range(self._w)] for y in range(self._h)], dtype=np.int32)
                
    def get_histogram(self, events):
        # Organizing the events in cells which are, then, saved as NumPy arrays.
        locmems = self._map_to_locmems(events)
        hist = np.zeros((self._ncells, self._npols, self._sdim, self._sdim), dtype=np.float32)
        # Now we have fun: we cycle on the local memories and generate the histogram corresponding to each of them.
        for c in range(self._ncells):
            for p in range(self._npols):
                hist[c,p,:,:] = np.sum(
                    np.stack(
                        [self._get_ts(locmems[c][p][i], locmems[c][p][:i]) for i in range(len(locmems[c][p]))]
                        if locmems[c][p].size>0
                        else
                        np.zeros((self._sdim, self._sdim), dtype=np.float32)
                    ),
                    axis=0
                )/max(1, locmems[c][p].size)
        return hist
                
    def _get_ts(self, event, locmem):
        t_i, t_j = event['t'].astype(np.float32), locmem['t'].astype(np.float32)
        # Starting time stamp, calculated subtracting the time window from the event timestamp.
        t_start = max(0, t_i-self._tw)
        # Relative coordinates in the time surfaces.
        ts_x, ts_y = locmem['x'] - event['x'], locmem['y'] - event['y']
        # Including only the events in the neighbourhood and in the time window. 
        mask = np.asarray((np.abs(ts_x)<=self._rho) & (np.abs(ts_y)<=self._rho) & (t_j>=t_start)).nonzero()[0]
        # For each event in the local memory that belongs to the spatial and temporal windows and for the current event, a time surface is generated.
        locmem_ts = np.zeros((1+len(mask), self._sdim, self._sdim), dtype=np.float32)
        if len(mask)>0:
            locmem_ts[np.arange(len(mask)), ts_y[mask]+self._rho, ts_x[mask]+self._rho] = np.exp(-(t_i-t_j[mask])/self._tau)
        # Adding the current event time surface.
        locmem_ts[-1, self._rho, self._rho] = 1
        # The accumulated time surfaces are returned.
        return np.sum(locmem_ts, axis=0)

    def _map_to_locmems(self, events):
        locmems = self._gen_locmems()
        # Mapping events to local memories. 
        for event in events:
            locmems[self._px_to_cell[int(event['x']), int(event['y'])]][max(0, int(event['p']))].append(event)
        # Converting the lists in structured NumPy arrays.
        locmems = [[np.stack(locmems[c][p]) if locmems[c][p] else np.empty((0,)) for p in range(self._npols)] for c in range(self._ncells)]
        return locmems
