#!/usr/bin/env python
# -- coding: utf-8 --
import numpy as np
import math as mt
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point
from tf.transformations  import euler_from_quaternion, quaternion_from_euler
import tf.transformations as tr

#def unamed():
#    return

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
        for i in range(_pbmat.shape[0]*_pbmat.shape[1]):
            idx = i/_pbmat.shape[1]
            idxf = mt.floor(idx)
            idy = i-idxf*_pbmat.shape[1]
            _pbmat[idxf,idy] = _pbmat[idxf,idy]*newmat[idxf,idy]
            
    def Ressample_particles(self):
        newPF = []
        finPF = [] #provisório
        for m in range(self.M):
            new_pos = 0#unamed()
            new_weight =0# unamed()
            newPF.append(Particle(m, new_pos, new_weight))
        for m in range(self.M):
            lin_pb = np.random.uniform(0,1,self.map.shape)
            self.particle_update_weight(lin_pb, newPF)
            mx = np.argmax(lin_pb)
            idx = mx/self.map.shape[1]
            idxf = mt.floor(idx)
            idy = mx-idxf*self.map.shape[1]
            finPF.append(Particle(m,[idxf,idy],1))    


def map_callback(msg):
    print (msg.info)


def get_odometry(self,msg):
    #print (msg.pose)

    # Pose

    self.robot_odom.pose.pose.position.x
    self.robot_odom.pose.pose.position.y
    self.robot_odom.pose.pose.position.z 

    # Orientation

    self.robot_odom.pose.pose.orientation.x
    self.robot_odom.pose.pose.orientation.y
    self.robot_odom.pose.pose.orientation.z
    self.robot_odom.pose.pose.orientation.w
    

def odom_processing(self,robot_odom):
    #Save robot Odometry


    # Determine the difference between new and old values
    self.last_robot_odom = self.robot_odom
    self.robot_odom = robot_odom

    if self.last_robot_odom:

        p_map_currbaselink = np.array([self.robot_odom.pose.pose.position.x,
                                        self.robot_odom.pose.pose.position.y,
                                        self.robot_odom.pose.pose.position.z])

        p_map_lastbaselink = np.array([self.last_robot_odom.pose.pose.position.x,
                                        self.last_robot_odom.pose.pose.position.y,
                                        self.last_robot_odom.pose.pose.position.z])

        q_map_lastbaselink = np.array([self.last_robot_odom.pose.pose.orientation.x,
                                        self.last_robot_odom.pose.pose.orientation.y,
                                        self.last_robot_odom.pose.pose.orientation.z,
                                        self.last_robot_odom.pose.pose.orientation.w])

        q_map_currbaselink = np.array([self.robot_odom.pose.pose.orientation.x,
                                        self.robot_odom.pose.pose.orientation.y,
                                        self.robot_odom.pose.pose.orientation.z,
                                        self.robot_odom.pose.pose.orientation.w])

        # Save quaternion units, with axis of rotation
        # Does the rotation matrix
        
        q_map_lastbaselink_euler = euler_from_quaternion(q_map_lastbaselink)
        q_map_currbaselink_euler = euler_from_quaternion(q_map_currbaselink)
        
        # Does the difference in yaw

        yaw_diff = q_map_currbaselink_euler[2] - q_map_lastbaselink_euler[2]


        self.dyaw += yaw_diff
        self.dx += p_lastbaselink_currbaselink[0]
        self.dy += p_lastbaselink_currbaselink[1]
def predict_next_odometry(self, particle):

    delta_x = random.gauss(0, self.dynamics_translation_noise_std_dev)
    delta_y = random.gauss(0, self.dynamics_translation_noise_std_dev)
    ntheta = random.gauss(0, self.dynamics_orientation_noise_std_dev)

    distance = sqrt(self.delta_x**2 + self.delta_y**2)


    particle.x += distance * cos(particle.theta) + nx
    particle.y += distance * sin(particle.theta) + ny
    particle.theta += self.dyaw + ntheta
   















def scan_callback(msg):
    print (msg.ranges) 


if __name__ == '__main__':
    mp = np.zeros((10,10))
    n_particles = 50
    d_range = [1,10]
    a_range = [-0.785398,2.356194]
    #PF = Particle_filter(n_particles, mp, d_range, a_range)
    # PF.Particle_init();
    rospy.init_node('Particle_Filter')
    #rospy.Subscriber('/mavros/local_position/odom', Odometry, odom_callback)
    #rospy.Subscriber('/base_scan', LaserScan, scan_callback)
    #rospy.Subscriber('/map', OccupancyGrid, map_callback)
    
    
    # Depois meter no init
    """ dynamics_translation_noise_std_dev   = 0.45
    dynamics_orientation_noise_std_dev   = 0.03
    beam_range_measurement_noise_std_dev = 0.3
     """
    
    rospy.spin()




