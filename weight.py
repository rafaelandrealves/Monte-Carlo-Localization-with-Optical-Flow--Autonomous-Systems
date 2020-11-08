#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:49:55 2020

@author: WhoKnows
"""

import numpy as np
import math as mt
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point

def unamed():
    return

class Particle(object):
    def __init__(self, _id, _pos, _weight, _theta=0):
        self.id = _id
        self.pos = _pos
        self.w = _weight
        self.theta = _theta
        
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
            lin_pb = np.random.uniform(0,1,(self.x,self.y))
            mx = np.argmax(lin_pb)
            idx = mx/self.y
            idxf = int(mt.floor(idx))
            idy = mx-idxf*self.y
            while (self.map[idx,idy] != 0):
                lin_pb[idx,idy] = 0
                mx = np.argmax(lin_pb)
                idx = mx/self.y
                idxf = int(mt.floor(idx))
                idy = mx-idxf*self.y
            self.particles.append(Particle(m,[idxf,idy],1))
    
    def particle_update_weight(self, _pbmat, _newPF):
        newmat = np.zeros((_pbmat.shape[0],_pbmat.shape[1]))
        for m in range(self.M):
            newmat[_newPF[m].pos[0], _newPF[m].pos[1]] = _newPF[m].w
        for i in range(_pbmat.shape[0]*_pbmat.shape[1]):
            idx = i/_pbmat.shape[1]
            idxf = mt.floor(idx)
            idy = i-idxf*_pbmat.shape[1]
            _pbmat[idxf,idy] = _pbmat[idxf,idy]*newmat[idxf,idy]
            
    def Resample_particles(self):
        newPF = []
        finPF = [] #provisório
        for m in range(self.M):
            new_pos = unamed()
            new_weight = unamed()
            newPF.append(Particle(m, new_pos, new_weight))
        for m in range(self.M):
            lin_pb = np.random.uniform(0,1,(self.x,self.y))
            self.particle_update_weight(lin_pb, newPF)
            mx = np.argmax(lin_pb)
            idx = mx/self.y
            idxf = int(mt.floor(idx))
            idy = mx-idxf*self.y
            while (self.map[idx,idy] != 0):
                lin_pb[idx,idy] = 0
                mx = np.argmax(lin_pb)
                idx = mx/self.y
                idxf = int(mt.floor(idx))
                idy = mx-idxf*self.y
            finPF.append(Particle(m,[idxf,idy],1))    
            
    def angle_vect_make(self, _max_angle, _min_angle, _angle_inc):
        n_values = int((_max_angle-_min_angle)/_angle_inc)
        self.angle_vector = np.zeros((1, n_values))
        self.angle_readings = n_values
        for i in range(n_values):
            self.angle_vector[i] = _min_angle+_angle_inc*i
            
    def map_resolve_size(self, _data):
        self.map = np.zeros((self.x,self.y))
        for i in range(self.x):
            for j in range(self.y):
                self.map[i,j] = _data(i*self.y+j)
    
    def m_to_grid(self):
        self.ranges_in_grid = np.zeros((2,self.angle_readings))
        for i in range(self.angle_readings):
            if self.ranges[i] < self.max_dist and self.ranges[i] > self.min_dist:
                self.ranges_in_grid[0,i] = (mt.cos(self.angle_vector[i]+(mt.pi/2))*self.ranges[i])/self.map_resolution
                self.ranges_in_grid[1,i] = (mt.sin(self.angle_vector[i]+(mt.pi/2))*self.ranges[i])/self.map_resolution
            else:
                self.ranges_in_grid[0,i] = -1
                self.ranges_in_grid[1,i] = -1
        
    def compare_dist(self, _m, _i):
        xx = int(mt.floor(self.ranges_in_grid[0,_i]))
        yy = int(mt.floor(self.ranges_in_grid[1,_i]))
        xi = self.particles[_m].pos[0]
        yi = self.particles[_m].pos[1]
        xw = xi+xx
        yw = yi+yy
        if self.map[xw,yw] != 0:
            self.particles[_m].w+=1
        
    def weight_change(self, _m):
        self.particles[_m].w = 1
        for i in range(self.angle_readings):
            if self.ranges_in_grid[0,i] != -1:
                self.compare_dist(_m,i)
        self.particles[_m].w = self.particles[_m].w/1023
        
    def scan_callback(self, msg):
        self.max_range_sensor = msg.range_max
        self.min_range_sensor = msg.range_min
        max_angle_sensor = msg.angle_max
        min_angle_sensor = msg.angle_min
        angle_inc_sensor = msg.angle_increment
        self.angle_vect_make(max_angle_sensor, min_angle_sensor, angle_inc_sensor)
        self.ranges = msg.ranges 
        
    def map_callback(self, msg):
        self.x = msg.info.width
        self.y = msg.info.height
        data = msg.data
        self.map_resolve_size(data)
        self.map_resolution = msg.info.resolution
        
def odom_callback(msg):
    print(msg.pose)   
        
if __name__ == '__main__':
    mp = np.zeros((10,10))
    n_particles = 50
    d_range = [1,10]
    a_range = [-0.785398,2.356194]
    PF = Particle_filter(n_particles, mp, d_range, a_range)
    PF.Particle_init();
    rospy.init_node('Particle_Filter')
    #rospy.Subscriber('/mavros/local_position/odom', Odometry, odom_callback)
    rospy.Subscriber('/base_scan', LaserScan, PF.scan_callback)
    rospy.Subscriber('/map', OccupancyGrid, PF.map_callback)
    rospy.spin()
