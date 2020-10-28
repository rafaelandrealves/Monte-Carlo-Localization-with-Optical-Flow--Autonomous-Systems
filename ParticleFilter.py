# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:49:55 2020

@author: marco
"""

import numpy as np
import math as mt

def Particle_init(_M, _map):
    dist = np.zeros(_map.shape)
    for m in range(_M):
        lin_pb = np.random.uniform(0,1,_map.shape)
        mx = np.argmax(lin_pb)
        idx = mx/_map.shape[1]
        idxf = mt.floor(idx)
        idc = mx-idxf*_map.shape[1]
        dist[idxf,idc] += 1
        lin_pb[idxf,idc] = 0
    return dist


mp = np.zeros((10,10))
PF = Particle_init(50, mp)
print(PF)