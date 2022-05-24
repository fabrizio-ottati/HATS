#! /usr/bin/env python 

import numpy as np
from math import ceil
from joblib import Parallel, delayed

class HATS():
    def __init__(self, sensor_size, tau, tw, cell_size, surface_size):
        self._w, self._h, self._npols = sensor_size
        self._tau, self._tw = np.array(tau, dtype=np.float32), np.array(tw, dtype=np.float32)
        self._cdim, self._sdim = cell_size, surface_size
        self._rho = surface_size//2
        self._hgrid, self._wgrid = ceil(self._h/self._cdim), ceil(self._w/self._cdim)
        self._ncells = self._hgrid*self._wgrid
        self._gen_locmems = lambda: [[[] for p in range(self._npols)] for c in range(self._ncells)]
        self._px_to_cell = np.array([[(y//self._cdim)*self._wgrid + x//self._cdim for x in range(self._w)] for y in range(self._h)], dtype=np.int32)
                
    def get_histogram(self, events):
        # Organizing the events in cells which are, then, saved as NumPy arrays.
        locmems = self._map_to_locmems(events)
        hist = np.zeros((self._ncells, self._npols, self._sdim, self._sdim), dtype=np.float32)
        # Now we have fun.
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
        t_start = max(0, t_i-self._tw)
        ts_x, ts_y = locmem['x'] - event['x'], locmem['y'] - event['y']
        # Including only the events in the neighbourhood and in the time window. 
        mask = np.asarray((np.abs(ts_x)<=self._rho) & (np.abs(ts_y)<=self._rho) & (t_j>=t_start)).nonzero()[0]
        ts = np.zeros((max(1,len(mask)), self._sdim, self._sdim), dtype=np.float32)
        if len(mask)>0:
            ts[np.arange(len(mask)), ts_y[mask]+self._rho, ts_x[mask]+self._rho] = np.exp(-(t_i-t_j[mask])/self._tau)
        ts[0, self._rho, self._rho] += 1
        return np.sum(ts, axis=0)

    def _map_to_locmems(self, events):
        locmems = self._gen_locmems()
        # Mapping events to local memories. 
        for event in events:
            locmems[self._px_to_cell[int(event['x']), int(event['y'])]][max(0, int(event['p']))].append(event)
        # Converting the lists in structured NumPy arrays.
        locmems = [[np.stack(locmems[c][p]) if locmems[c][p] else np.empty((0,)) for p in range(self._npols)] for c in range(self._ncells)]
        return locmems
