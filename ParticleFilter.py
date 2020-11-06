# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:49:55 2020

@author: WhoKnows
"""

import numpy as np
import math as mt

def unamed():
    return

class Particle(object):
    def __init__(self, _id, _pos, _weight):
        self.id = _id
        self.pos = _pos
        self.w = _weight
        
class Particle_filter(object):
    def __init__(self, _M, _map, _distance_range, _angle_range):
        self.M = _M
        self.map = _map
        self.x = _map.shape[0]
        self.y = _map.shape[1]
        self.min_dist = _distance_range[0]
        self.max_dist = _distance_range[1]
        self.min_ang = _angle_range[0]
        self.max_ang = _angle_range[1]
        self.particles = []
        
    def Particle_init(self):
        for m in range(self.M):
            lin_pb = np.random.uniform(0,1,self.map.shape)
            mx = np.argmax(lin_pb)
            idx = mx/self.map.shape[1]
            idxf = mt.floor(idx)
            idy = mx-idxf*self.map.shape[1]
            self.particles.append(Particle(m,[idxf,idy],1))
    
    def particle_update_weight(self, _pbmat, _newPF):
        newmat = np.zeros((_pbmat.shape[0],_pbmat.shape[1]))
        for m in range(self.M):
            newmat[_newPF[m].pos[0], _newPF[m].pos[1]] = _newPF[m].w
        for i in range(_pbmat.shape[0]*_pbmat[1]):
            idx = i/_pbmat.shape[1]
            idxf = mt.floor(idx)
            idy = i-idxf*_pbmat.shape[1]
            _newPF[idxf,idy] = _newPF[idxf,idy]*newmat[idxf,idy]
            
    def Ressample_particles(self):
        newPF = []
        finPF = [] #provisório
        for m in range(self.M):
            new_pos = unamed()
            new_weight = unamed()
            newPF.append(Particle(m, new_pos, new_weight))
        for m in range(self.M):
            lin_pb = np.random.uniform(0,1,self.map.shape)
            self.particle_update_weight(lin_pb, newPF)
            mx = np.argmax(lin_pb)
            idx = mx/self.map.shape[1]
            idxf = mt.floor(idx)
            idy = mx-idxf*self.map.shape[1]
            finPF.append(Particle(m,[idxf,idy],1))    
        
if __name__ == '__main__':
    mp = np.zeros((10,10))
    n_particles = 50
    d_range = [1,10]
    a_range = [-0.785398,2.356194]
    PF = Particle_filter(n_particles, mp, d_range, a_range)
    PF.Particle_init();
