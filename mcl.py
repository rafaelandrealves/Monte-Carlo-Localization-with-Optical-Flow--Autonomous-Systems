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
import random
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Header, ColorRGBA
from threading import Thread, Lock

def id_array(_size):
    new_array = np.zeros(_size)
    for i in range(_size):
        new_array[i] = i
    return new_array

class Particle(object):
    def __init__(self, _id, _pos, _weight, _theta=0):
        self.id = _id
        self.pos = _pos
        self.w = _weight
        self.theta = _theta

class Particle_filter(object):
    def __init__(self, _M, _dynamics_translation_noise_std_dev,
                 _dynamics_orientation_noise_std_dev,
                 _beam_range_measurement_noise_std_dev,
                 _distance_range = [0,0], _angle_range = [0,0]):
        self.M = _M
        self.map = []
        self.x = 0
        self.y = 0
        self.min_dist = _distance_range[0]
        self.max_dist = _distance_range[1]
        self.min_ang = _angle_range[0]
        self.max_ang = _angle_range[1]
        self.map_resolution = 0
        self.particles = []
        self.dynamics_orientation_noise_std_dev = _dynamics_orientation_noise_std_dev
        self.dynamics_translation_noise_std_dev = _dynamics_translation_noise_std_dev
        self.beam_range_measurement_noise_std_dev = _beam_range_measurement_noise_std_dev


        # Previous odometry measurement of the robot
        self.last_robot_odom = None

        # Current odometry measurement of the robot
        self.robot_odom = None

        # Angle vector/readings

        self.angle_vector = []
        self.angle_readings = 0

        # Init ranges of ogm

        self.ranges_in_grid = []
        self.ranges = []
        # variance of values in point coordinates

        self.dyaw = 0
        self.dx = 0
        self.dy = 0

    def Particle_init(self):
        for m in range(self.M):
            while self.x == 0:
                rospy.sleep(1)
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
            newmat[int(_newPF[m].pos[0]), int(_newPF[m].pos[1])] = _newPF[m].w
        for i in range(_pbmat.shape[0]*_pbmat.shape[1]):
            idx = i/_pbmat.shape[1]
            idxf = int(mt.floor(idx))
            idy = i-idxf*_pbmat.shape[1]
            _pbmat[idxf,idy] = _pbmat[idxf,idy]*newmat[idxf,idy]
        return _pbmat

    def det_eff_part(self, _w_vect): #criado
        sum_weight = 0
        for m in range(self.M):
            sum_weight = sum_weight + _w_vect[m]**2
        n_eff = 1/(sum_weight)
        return n_eff

    def normalize_weights(self, _n_w): #criado
        total_w = sum(_n_w)
        for m in range(self.M):
            _n_w[m] = float(_n_w[m])/total_w
        return _n_w

    def make_1D_vect(self, _w):
        id_vect = id_array(self.x*self.y)
        w_vect = np.zeros(self.x*self.y)
        for m in range(self.M):
            mx = self.particles[m].pos[0]*self.y+self.particles[m].pos[1]
            mx = int(mx)
            w_vect[mx] = _w[m]
        return id_vect, w_vect

    def Resample_particles(self):
        newPF = []
        finPF = [] #provisório
        new_weight = np.zeros(self.M) #alterado
        self.m_to_grid()
        for m in range(self.M):
            [new_pos,new_theta] = self.predict_next_odometry(m) #alterado
            if self.particles[m].w != 0:
                new_weight[m] = self.weight_change(m) #alterado
            if m == 1:
                print(new_pos, new_theta, new_weight[m])
            newPF.append(Particle(m, new_pos, new_weight[m], _theta = new_theta)) #alterado
        new_weight = self.normalize_weights(new_weight)
        eff_particles = self.det_eff_part(new_weight) #alterado
        #print(new_weight)
        #print('EFfective particle:',eff_particles)
        if eff_particles < (self.M)/4:
            for m in range(self.M):
                a, w_v = self.make_1D_vect(new_weight)
                w_v /= sum(w_v)
                mx = np.random.choice(a, p=w_v)
                idx = mx/self.y
                idxf = int(mt.floor(idx))
                idy = mx-idxf*self.y
                finPF.append(Particle(m,[idxf,idy],1, _theta = self.particles[m].theta))
            self.particles = finPF

    def angle_vect_make(self, _max_angle, _min_angle, _angle_inc):
        n_values = int((_max_angle-_min_angle)/_angle_inc)
        self.angle_vector = np.zeros((n_values,1))
        self.angle_readings = n_values
        for i in range(n_values):
            self.angle_vector[i] = _min_angle+_angle_inc*i

    def map_resolve_size(self, _data):
        self.map = np.zeros((self.x,self.y))
        for i in range(self.x):
            for j in range(self.y):
                self.map[i,j] = _data[i*self.y+j]

    def m_to_grid(self):
        self.ranges_in_grid = np.zeros((2,self.angle_readings))
        for i in range(self.angle_readings):
            if self.ranges[i] < self.max_dist and self.ranges[i] > self.min_dist:
                self.ranges_in_grid[0,i] = self.ranges[i]/self.map_resolution
                self.ranges_in_grid[1,i] = self.ranges[i]/self.map_resolution
            else:
                self.ranges_in_grid[0,i] = -1
                self.ranges_in_grid[1,i] = -1

    def subsample(self, _msg):
        subsample_range = []
        subsample_angle = []
        for i in range(self.angle_readings):
            if self.ranges[i] < self.max_dist and self.ranges[i] > self.min_dist:
                subsample_angle.append(self.angle_vector[i])
                subsample_range.append(self.ranges[i])
        return subsample_range, subsample_angle

    def compare_dist(self, _m, _i, _wt):
        ang_dist_x = mt.cos(self.angle_vector[_i]+(mt.pi/2)+self.particles[_m].theta)*self.ranges_in_grid[0,_i]
        ang_dist_y = mt.sin(self.angle_vector[_i]+(mt.pi/2)+self.particles[_m].theta)*self.ranges_in_grid[1,_i]
        xx = int(mt.floor(ang_dist_x))
        yy = int(mt.floor(ang_dist_y))
        xi = int(self.particles[_m].pos[0])
        yi = int(self.particles[_m].pos[1])
        xw = xi+xx
        yw = yi+yy
        wa = 0
        for i in range(-2,2):
            if(xw+i > 0 and xw+i < self.x and yw+i > 0 and yw+i < self.y):
                wa = wa + self.map[xw+i,yw+i]
        if wa > 0:
            return 1
        else:
            return 0

    def weight_change(self, _m):
        wt = 1
        for i in range(self.angle_readings):
            if self.ranges_in_grid[0,i] != -1:
                wt = wt + self.compare_dist(_m,i,wt)
        return wt

    def odometry_correct(self, _m):
        xx = int(self.particles[_m].pos[0])
        yy = int(self.particles[_m].pos[1])
        if xx >= self.x or xx < 0 or yy >= self.y or yy < 0:
            self.particles[_m].w = 0
            return
        if self.map[xx,yy] != 0:
            self.particles[_m].w = 0

    def odom_processing(self,msg):
        #Save robot Odometry


        # Determine the difference between new and old values
        self.last_robot_odom = self.robot_odom
        self.robot_odom = msg

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
            R_map_lastbaselink = tr.quaternion_matrix(q_map_lastbaselink)[0:3,0:3]

            p_lastbaselink_currbaselink = R_map_lastbaselink.transpose().dot(p_map_currbaselink - p_map_lastbaselink)

            q_map_lastbaselink_euler = euler_from_quaternion(q_map_lastbaselink)
            q_map_currbaselink_euler = euler_from_quaternion(q_map_currbaselink)

            # Does the difference in yaw

            yaw_diff = q_map_currbaselink_euler[2] - q_map_lastbaselink_euler[2]


            self.dyaw += yaw_diff
            self.dx += p_lastbaselink_currbaselink[0]
            self.dy += p_lastbaselink_currbaselink[1]


    def predict_next_odometry(self, m):

        delta_x = random.gauss(0, self.dynamics_translation_noise_std_dev)
        delta_y = random.gauss(0, self.dynamics_translation_noise_std_dev)
        ntheta = random.gauss(0, self.dynamics_orientation_noise_std_dev)

        distance = mt.sqrt(self.dx**2 + self.dy**2)


        self.particles[m].pos[0] += distance * mt.cos(self.particles[m].theta) + delta_x
        self.particles[m].pos[1] += distance * mt.sin(self.particles[m].theta) + delta_y
        self.particles[m].theta += self.dyaw + ntheta
        #print('The particle',m,'is in (',self.particles[m].pos[0],',',self.particles[m].pos[1],')')
        self.odometry_correct(m)
        return [[self.particles[m].pos[0],self.particles[m].pos[1]],self.particles[m].theta]

    def scan_analysis(self, msg):
        max_angle_sensor = msg.angle_max
        min_angle_sensor = msg.angle_min
        angle_inc_sensor = msg.angle_increment
        self.max_dist = msg.range_max
        self.min_dist = msg.range_min
        self.angle_vect_make(max_angle_sensor, min_angle_sensor, angle_inc_sensor)
        self.ranges = msg.ranges

    def get_map(self, msg):
        self.x = msg.info.width
        self.y = msg.info.height
        data = msg.data
        self.map_resolve_size(data)
        self.map_resolution = msg.info.resolution



class MCL(object):
    def __init__(self,num_particles):

        # Init node

        rospy.init_node('Monte_Carlo')

        # Errors of associated devices

        dynamics_translation_noise_std_dev   = 0.45
        dynamics_orientation_noise_std_dev   = 0.03
        beam_range_measurement_noise_std_dev = 0.3

        # Get the Particle Filter running

        self.pf = Particle_filter(num_particles,
                                 dynamics_translation_noise_std_dev,
                                 dynamics_orientation_noise_std_dev,
                                 beam_range_measurement_noise_std_dev)

        # Get MAP

        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

        self.pf.Particle_init()
        self.mutex = Lock()


        self.laser_points_marker_pub = rospy.Publisher('/typhoon/debug/laser_points', Marker, queue_size=1)
        self.particles_pub = rospy.Publisher('/typhoon/particle_filter/particles', MarkerArray, queue_size=1)
        # Subscribe to topics
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/base_scan', LaserScan, self.scan_callback)




    def scan_callback(self,msg):
        self.mutex.acquire()
        self.publish_laser_pts(msg)
        self.pf.scan_analysis(msg)
        self.pf.Resample_particles()
        self.pf.dx = 0
        self.pf.dy = 0
        self.pf.dyaw = 0
        self.mutex.release()


    def map_callback(self, msg):
        self.pf.get_map(msg)

    def odom_callback(self, msg):
        self.mutex.acquire()
        self.pf.odom_processing(msg)
        self.mutex.release()


    # def get_2d_laser_points_marker(self, timestamp, frame_id, pts_in_map, marker_id, rgba):
    #     msg = Marker()
    #     msg.header.stamp = timestamp
    #     msg.header.frame_id = frame_id
    #     msg.ns = 'laser_points'
    #     msg.id = marker_id
    #     msg.type = 6
    #     msg.action = 0
    #     msg.points = [Point(pt[0], pt[1], pt[2]) for pt in pts_in_map]
    #     msg.colors = [rgba for pt in pts_in_map]
    #
    #     for pt in pts_in_map:
    #         assert((not np.isnan(pt).any()) and np.isfinite(pt).all())
    #
    #     msg.scale.x = 0.1
    #     msg.scale.y = 0.1
    #     msg.scale.z = 0.1
    #     return msg

    def publish_laser_pts(self, msg):
        """Publishes the currently received laser scan points from the robot, after we subsampled
        them in order to comparse them with the expected laser scan from each particle."""
        if self.pf.robot_odom is None:
            return

        subsampled_ranges, subsampled_angles = self.pf.subsample(msg)


        N = len(subsampled_ranges)
        x = self.pf.robot_odom.pose.pose.position.x
        y = self.pf.robot_odom.pose.pose.position.y
        _, _ , yaw_in_map = tr.euler_from_quaternion(np.array([self.pf.robot_odom.pose.pose.orientation.x,
                                                               self.pf.robot_odom.pose.pose.orientation.y,
                                                               self.pf.robot_odom.pose.pose.orientation.z,
                                                               self.pf.robot_odom.pose.pose.orientation.w]))

        pts_in_map = [ (x ,
                        y ,
                        self.pf.map_resolution) for r,theta in zip(subsampled_ranges, subsampled_angles)]


        # lpmarker = self.get_2d_laser_points_marker(msg.header.stamp, 'map', pts_in_map, 30000, ColorRGBA(0, 0.0, 0, 1.0))
        # self.laser_points_marker_pub.publish(lpmarker)


    def get_particle_marker(self, timestamp, particle, marker_id):
        """Returns an rviz marker that visualizes a single particle"""
        msg = Marker()
        msg.header.stamp = timestamp
        msg.header.frame_id = 'map'
        msg.ns = 'particles'
        msg.id = marker_id
        msg.type = 0  # arrow
        msg.action = 0 # add/modify
        msg.lifetime = rospy.Duration(1)

        yaw_in_map = particle.theta
        vx = mt.cos(yaw_in_map)
        vy = mt.sin(yaw_in_map)
        msg.color = ColorRGBA(0, 1.0, 0, 1.0)
        if marker_id == 5:
            msg.color = ColorRGBA(1.0, 0.0, 0, 1.0)

        #print('This is Particle x-',particle.pos[0])
        #print('This is Particle y-',particle.pos[1])

        gx = particle.pos[0] * self.pf.map_resolution - 10
        gy = particle.pos[1] * self.pf.map_resolution - 10
        msg.points.append(Point(gx, gy, particle.theta))
        msg.points.append(Point(gx + self.pf.map_resolution*vx, gy + self.pf.map_resolution*vy, particle.theta))

        msg.scale.x = 0.05
        msg.scale.y = 0.15
        msg.scale.z = 0.1
        return msg

    def publish_particle_markers(self):
        """ Publishes the particles of the particle filter in rviz"""
        ma = MarkerArray()
        ts = rospy.Time.now()
        for i in xrange(len(self.pf.particles)):
            ma.markers.append(self.get_particle_marker(ts, self.pf.particles[i], i))

        self.particles_pub.publish(ma)

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.publish_particle_markers()
            rate.sleep()




if __name__ == '__main__':

    numero_particulas = 100

    Monte_carlo = MCL(numero_particulas)

    Monte_carlo.run()

    #rospy.spin()
