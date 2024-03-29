#!/usr/bin/env python
# -- coding: utf-8 --
import numpy as np
import math as mt
import rospy
import operator
from mavros_msgs.msg import OpticalFlowRad, Altitude
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point, Pose, Quaternion, PoseArray
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
    def __init__(self, _id, _pos, _weight, _theta=0, _w_theta = 1):
        self.id = _id
        self.pos = _pos
        self.w = _weight
        self.theta = _theta
        self.w_theta = _w_theta

class Particle_filter(object):
    def __init__(self, _M, _dynamics_translation_noise_std_dev,
                 _dynamics_orientation_noise_std_dev,
                 _beam_range_measurement_noise_std_dev,
                 _distance_range = [0.5,0], _angle_range = [0,0]):
        self.M = _M
        self.map = []
        self.x = 0
        self.y = 0
        self.origin_x = 0
        self.origin_y = 0
        self.min_dist = _distance_range[0]
        self.max_dist = _distance_range[1]
        self.min_ang = _angle_range[0]
        self.max_ang = _angle_range[1]
        self.map_resolution = 0
        self.particles = []
        self.dynamics_orientation_noise_std_dev = _dynamics_orientation_noise_std_dev
        self.dynamics_translation_noise_std_dev = _dynamics_translation_noise_std_dev
        self.beam_range_measurement_noise_std_dev = _beam_range_measurement_noise_std_dev
        self.i = 0

        # Previous odometry measurement of the robot
        self.last_robot_odom = None

        # Current odometry measurement of the robot
        self.robot_odom = None

        # Angle vector/readings

        self.angle_vector = []
        self.angle_readings = 0

        # Init ranges of ogm
        self.first_time = True
        self.ranges_in_grid = []
        self.ranges = []
        self.ranges_temp = []
        # variance of values in point coordinates

        self.total_readings = 0
        self.dyaw_temp = 0
        self.dx_temp = 0
        self.dy_temp = 0
        self.dyaw = 0
        self.dx = 0
        self.dy = 0
        self.ground_truth_x_now = 0
        self.ground_truth_y_now = 0
        self.ground_truth_yaw_now = 0
        self.x_p = 0
        self.y_p = 0
        self.theta_p = 0
        self.old_theta = 0
        self.girox = 0
        self.giroy = 0
        self.ground_truth_x = 0
        self.ground_truth_y = 0
        self.ground_truth_yaw = 0


        self.error_file_position = []
        self.error_file_orientation = []
        self.occupancy_file = []
        self.particle_pos_file = []
        self.ground_truth_file = []
        self.prediction_file = []
        self.curr_odometry = None
        self.imu_velocity = None
        self.new_velocity = None
        self.Vx_temp = 0
        self.Vy_temp = 0
        self.vx = 0
        self.vy = 0
        self.time = 0
        self.last_time = 0
        self.dif_t = 0
        self.dif_time = 0
        self.last_velocity = None
        self.curr_velocity = None
        self.imu_velocity = None
        self.velocity_x = 0
        self.velocity_y = 0
        self.opt_velocity = None
        self.altitude = 0
        self.last_orientation = None
        self.curr_orientation = None
        self.ang = 0
        self.angx = 0
        self.roll = 0
        self.pitch = 0
        self.droll = 0
        self.dpitch = 0
        self.forward_x = 0
        self.side_y = 0
        self.yaw_now = 0
        self.yaw = 0


    def Particle_init(self):
        for m in range(self.M):
            while self.x == 0: #while map not received
                rospy.sleep(1)
            lin_pb = np.random.uniform(0,1,(self.x,self.y)) #create uniform distribution matrix with map size
            mx = np.argmax(lin_pb) #max probability from matrix
            idx = mx/self.y #index x from the max
            idxf = int(mt.floor(idx)) # index x normalized
            idy = mx-idxf*self.y #index y from the max
            while(self.map[idxf,idy] != 0): # search for valid position
                lin_pb[idxf,idy] = 0
                mx = np.argmax(lin_pb)
                idx = mx/self.y
                idxf = int(mt.floor(idx))
                idy = mx-idxf*self.y
            theta_t = np.random.uniform()*2*mt.pi-mt.pi
            # if(m==5):
            #     self.particles.append(Particle(m,[422,384],1))
            #     print('map=',self.map[422,384])
            # else:
            self.particles.append(Particle(m,[idxf,idy],1, _theta = theta_t)) # append particle to particle filter
            #print('map=',self.map[idxf,idy])

    def Init_one_particle(self, _m):
        ntheta = random.gauss(0, 1) #create gauss
        lin_pb = np.random.uniform(0,1,(self.x,self.y)) #create uniform distribution matrix with map size
        mx = np.argmax(lin_pb) #max probability from matrix
        idx = mx/self.y #index x from the max
        idxf = int(mt.floor(idx)) # index x normalized
        idy = mx-idxf*self.y #index y from the max
        while(self.map[idxf,idy] != 0): # search for valid position
            lin_pb[idxf,idy] = 0
            mx = np.argmax(lin_pb)
            idx = mx/self.y
            idxf = int(mt.floor(idx))
            idy = mx-idxf*self.y
        self.particles[_m].pos[0] = idxf
        self.particles[_m].pos[1] = idy
        self.particles[_m].w = 1/self.M
        self.particles[_m].theta = np.random.uniform()*2*mt.pi-mt.pi#uncomment for kidnap

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

    def Initar_init(self):
        while self.ground_truth_x == 0: #while not started
            rospy.sleep(1)
        for m in range(self.M):
            self.particles[m].pos[0] = 150# self.ground_truth_x
            self.particles[m].pos[1] = 100#self.ground_truth_y
            self.particles[m].theta = mt.pi/2#self.ground_truth_yaw

    def get_ground_truth(self, msg):
        while self.map_resolution == 0: #while not started
            rospy.sleep(1)
        p_map_currbaselink = np.array([msg.pose.pose.position.x,
                                       msg.pose.pose.position.y,
                                       msg.pose.pose.position.z])

        q_map_currbaselink = np.array([msg.pose.pose.orientation.x,
                                       msg.pose.pose.orientation.y,
                                       msg.pose.pose.orientation.z,
                                       msg.pose.pose.orientation.w])

        self.ground_truth_x = (p_map_currbaselink[0] - self.origin_x)/self.map_resolution
        self.ground_truth_y = (p_map_currbaselink[1] - self.origin_y)/self.map_resolution
        q_map_currbaselink_euler = euler_from_quaternion(q_map_currbaselink)
        self.ground_truth_yaw = q_map_currbaselink_euler[2]

    def det_eff_part(self, _w_vect):
        sum_weight = 0
        for m in range(self.M):
            sum_weight = sum_weight + _w_vect[m]**2
        n_eff = 1/(sum_weight)
        return n_eff

    def normalize_weights(self, _n_w):
        total_w = sum(_n_w) #total weight
        for m in range(self.M):
            _n_w[m] = float(_n_w[m])/total_w #normalized
        return _n_w

    def make_1D_vect(self, _w):
        id_vect = self.x*self.y #max
        w_vect = np.zeros(self.x*self.y) #weight vector
        for m in range(self.M):
            if self.particles[m].pos[0] < self.x and self.particles[m].pos[0] >= 0 and self.particles[m].pos[1] < self.y and self.particles[m].pos[1] >= 0:
                mx = self.particles[m].pos[0]*self.y+self.particles[m].pos[1] #create index
                mx = int(mx)
                w_vect[mx] += _w[m]
        return id_vect, w_vect

    def Pos_predict(self, _w_v, _w_t):
        i = 0
        temp_weight = _w_v
        x_pred = 0
        y_pred = 0
        theta_pred = 0
        for i in range(self.M):
            x_pred += self.particles[i].pos[0]*_w_v[i]
            y_pred += self.particles[i].pos[1]*_w_v[i]
            theta_pred += self.particles[i].theta*_w_t[i]
        # Nm = int(0.1 * self.M)
        # d = {k:v for k, v in enumerate(temp_weight)}
        # sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        # id = [k[0] for k in sorted_d][:Nm]
        # while i < Nm:
        #     #id = np.argmax(temp_weight)
        #     x_pred += self.particles[id[i]].pos[0]
        #     y_pred += self.particles[id[i]].pos[1]
        #     theta_pred += self.particles[id[i]].theta
        #     #temp_weight[id] = 0
        #     i+=1
        # x_pred = x_pred / Nm
        # y_pred = y_pred / Nm
        # theta_pred = theta_pred / Nm
        if theta_pred > mt.pi:
            theta_pred -= 2*mt.pi
        elif theta_pred < -(mt.pi):
            theta_pred += 2*mt.pi
        # for m in range(self.M):
        #     self.particles[m].theta = theta_pred
        return x_pred, y_pred, theta_pred

    def error_calc(self):
        #print('prediction',self.x_p,' ', self.y_p, ' ', self.theta_p)
        #print('\nreality',self.ground_truth_x_now,' ', self.ground_truth_y_now, ' ', self.ground_truth_yaw_now)
        error_pos = mt.sqrt(((self.x_p-self.ground_truth_x_now)*self.map_resolution)**2 +
                        ((self.y_p-self.ground_truth_y_now)*self.map_resolution)**2)
        error_ori = self.theta_p-self.ground_truth_yaw_now
        return error_pos, error_ori

    def check_divergence(self, _new_weight):
        max_w = max(_new_weight)
        rate_w = max_w / self.total_readings
        if rate_w < 0.70:
            i = 0
            Nm = int(0.1 * self.M)
            d = {k:v for k, v in enumerate(_new_weight)}
            sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=False)
            id = [k[0] for k in sorted_d][:Nm]
            while i < Nm:
                #id = np.argmax(temp_weight)
                self.Init_one_particle(id[i])
                #temp_weight[id] = 0
                i+=1

    def Resample_particles(self):
        while self.max_dist == 0: #while not started
            rospy.sleep(1)
        self.ranges = self.ranges_temp #assign to local variable
        self.dyaw = self.dyaw_temp
        if self.forward_x < 0:
            self.angx = mt.pi
        if self.side_y > 0:
            self.ang = mt.pi/2
        else:
            self.ang = -mt.pi/2
        self.dy = abs(self.dy_temp)
        self.dx = abs(self.dx_temp)
        self.angx = 0
        self.dyaw_temp = 0
        self.dy_temp = 0
        self.dx_temp = 0
        self.velocity_x = 0
        self.velocity_y = 0
        self.yaw_now = self.yaw
        #print('Pitch',self.pitch,'Roll-', self.roll)
        self.roll = 0
        self.pitch = 0
        self.forward_x = 0
        self.side_y = 0
        self.ground_truth_x_now = self.ground_truth_x
        self.ground_truth_y_now = self.ground_truth_y
        self.ground_truth_yaw_now = self.ground_truth_yaw
        self.ground_truth_file.write(repr(self.ground_truth_x_now)+ ' ' +repr(self.ground_truth_y_now)+ ' ' +repr(self.ground_truth_yaw_now)+'\n')
        newPF = [] # X~
        finPF = [] # X
        new_weight = np.zeros(self.M) #new vector for weight change
        new_weight_theta = np.zeros(self.M) #new vector for weight change
        #rospy.sleep(10)
        error = 0
        self.m_to_grid() # convert from meters to grid coordinates
        for m in range(self.M):
            [new_pos,new_theta] = self.predict_next_odometry(m) #predict odom
            new_weight[m] = self.weight_change(m) # predict weight
            new_weight_theta[m] = self.theta_weight(m) # predict theta
            self.particles[m].w = new_weight[m] # assign to particle
            newPF.append(Particle(m, new_pos, new_weight[m], _theta = new_theta)) #create new PF
        self.check_divergence(new_weight) #comment in order to do tracking
        new_weight = self.normalize_weights(new_weight)
        new_weight_theta = self.normalize_weights(new_weight_theta)
        eff_particles = self.det_eff_part(new_weight)
        self.x_p, self.y_p, self.theta_p = self.Pos_predict( new_weight, new_weight_theta)
        self.prediction_file.write(repr(self.x_p)+ ' ' +repr(self.y_p)+ ' ' +repr(self.theta_p)+'\n')
        #print('A particula 2 esta na posica x:',self.particles[2].pos[0],'e y:',self.particles[2].pos[1],'\n')
        #print(new_weight)
        #print('EFfective particle:',eff_particles)
        #print('Error', (mt.sqrt(error)/self.M ))
        error_position, error_orientation = self.error_calc()
        self.error_file_position.write(repr(error_position)+'\n')
        self.error_file_orientation.write(repr(error_orientation)+'\n')
        print('Error', error_position,error_orientation)
        b_idx = np.argmax(new_weight) #get best prediction index
        max_weight = max(new_weight) #get best prediction
        #print('The best estimate is given by x = ', self.particles[b_idx].pos[0]*self.map_resolution,' and y = ', self.particles[b_idx].pos[1]*self.map_resolution,' with weight = ', max_weight)
        if eff_particles < 2*(self.M)/3:
            w_v = np.array(new_weight)
            w_v = w_v*self.M
            w_v_t = np.array(new_weight_theta)
            w_v_t = w_v_t*self.M
            for m in range(self.M):
                nposx = random.gauss(0, self.beam_range_measurement_noise_std_dev) #create gauss
                nposy = random.gauss(0, self.beam_range_measurement_noise_std_dev) #create gauss
                if(max(w_v)<1):
                    mx = np.argmax(w_v)
                    w_v[mx] = 0
                else:
                    mx = np.argmax(w_v) #max probability from matrix
                    w_v[mx] = w_v[mx] - 1
                idx = self.particles[mx].pos[0]
                idy = self.particles[mx].pos[1]
                if(max(w_v_t)<1):
                    mx_t = np.argmax(w_v_t)
                    w_v_t[mx_t] = 0
                else:
                    mx_t = np.argmax(w_v_t) #max probability from matrix
                    w_v_t[mx_t] = w_v_t[mx_t] - 1
                idt = self.particles[mx_t].theta
                finPF.append(Particle(m,[idx+nposx,idy+nposy],1, _theta = idt))
            self.particles = finPF

    def angle_vect_make(self, _max_angle, _min_angle, _angle_inc):
        n_values = int((_max_angle-_min_angle)/_angle_inc) #number of readings
        self.angle_vector = np.zeros((n_values,1)) #intialize vector
        self.angle_readings = n_values
        for i in range(n_values):
            self.angle_vector[i] = _min_angle+_angle_inc*i #assign the angle to the vector position

    def map_resolve_size(self, _data):
        self.map = np.zeros((self.x,self.y))
        for i in range(self.x):
            for j in range(self.y):
                self.map[i,j] = _data[j*self.x+i]
                self.occupancy_file.write(repr(self.map[i,j])+ ' ')
            self.occupancy_file.write('\n')

    def m_to_grid(self):
        self.ranges_in_grid = np.zeros((2,self.angle_readings)) #create new ranges vector
        self.total_readings = 0
        for i in range(self.angle_readings):
            if self.ranges[i] < self.max_dist and self.ranges[i] > self.min_dist: # in range
                self.ranges_in_grid[0,i] = self.ranges[i]/self.map_resolution #convert from meters to grid
                self.ranges_in_grid[1,i] = self.ranges[i]/self.map_resolution #convert from meters to grid
                self.total_readings += 1
            else: # not valid
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

    def compare_dist(self, _m, _i):
        ang_dist_x = mt.cos(self.angle_vector[_i]+self.particles[_m].theta+mt.pi/2)*self.ranges_in_grid[0,_i] #
        ang_dist_y = mt.sin(self.angle_vector[_i]+self.particles[_m].theta+mt.pi/2)*self.ranges_in_grid[1,_i] #trigonometry
        xx = int(mt.floor(ang_dist_x))
        yy = int(mt.floor(ang_dist_y))
        xi = int(self.particles[_m].pos[0])
        yi = int(self.particles[_m].pos[1])
        xw = xi+xx
        yw = yi+yy
        wa = 0
        for i in range(-1,2):
            if(xw+i >= 0 and xw+i < self.x and yw+i >= 0 and yw+i < self.y):
                wa = wa + self.map[xw+i,yw+i]
        if wa > 0:
            return 1
        else:
            return 0


    def theta_weight(self,_m):
        norm_error = abs(self.particles[_m].theta - self.yaw)
        prob_theta = np.exp(-0.5 * (1/1.2) * norm_error**2)

        return prob_theta

    def weight_change(self, _m):
        wt = 1 # temporary weight
        for i in range(self.angle_readings): # for all laser readings
            if self.ranges_in_grid[0,i] != -1: # check if is valid
                wt = wt + self.compare_dist(_m,i) # change weight
        #wt = wt / self.total_readings
        return wt

    def odometry_correct(self, _m):
        xx = int(self.particles[_m].pos[0]) # pos x from particle
        yy = int(self.particles[_m].pos[1]) # pos y from particle
        if xx >= self.x or xx < 0 or yy >= self.y or yy < 0: #check if it is outside map
            self.Init_one_particle(_m)
            return
        if self.map[xx,yy] != 0: #check if it is in available place
            self.Init_one_particle(_m)

    # def odom_processing(self,msg):
    #     #Save robot Odometry
    #
    #     # Determine the difference between new and old values
    #     self.last_robot_odom = self.robot_odom
    #     self.robot_odom = msg
    #
    #     if self.last_robot_odom: #if its not the first time
    #
    #         p_map_currbaselink = np.array([self.robot_odom.pose.pose.position.x,
    #                                         self.robot_odom.pose.pose.position.y,
    #                                         self.robot_odom.pose.pose.position.z])
    #
    #         p_map_lastbaselink = np.array([self.last_robot_odom.pose.pose.position.x,
    #                                         self.last_robot_odom.pose.pose.position.y,
    #                                         self.last_robot_odom.pose.pose.position.z])
    #
    #         q_map_lastbaselink = np.array([self.last_robot_odom.pose.pose.orientation.x,
    #                                         self.last_robot_odom.pose.pose.orientation.y,
    #                                         self.last_robot_odom.pose.pose.orientation.z,
    #                                         self.last_robot_odom.pose.pose.orientation.w])
    #
    #         q_map_currbaselink = np.array([self.robot_odom.pose.pose.orientation.x,
    #                                         self.robot_odom.pose.pose.orientation.y,
    #                                         self.robot_odom.pose.pose.orientation.z,
    #                                         self.robot_odom.pose.pose.orientation.w])
    #
    #         # Save quaternion units, with axis of rotation
    #         # Does the rotation matrix
    #         p_lastbaselink_currbaselink = p_map_currbaselink-p_map_lastbaselink
    #         # R_map_lastbaselink = tr.quaternion_matrix(q_map_lastbaselink)[0:3,0:3]
    #         #
    #         # p_lastbaselink_currbaselink = R_map_lastbaselink.transpose().dot(p_map_currbaselink - p_map_lastbaselink)
    #
    #         q_map_lastbaselink_euler = euler_from_quaternion(q_map_lastbaselink)
    #         q_map_currbaselink_euler = euler_from_quaternion(q_map_currbaselink)
    #         # q_lastbaselink_currbaselink = tr.quaternion_multiply(tr.quaternion_inverse(q_map_lastbaselink), q_map_currbaselink)
    #         #
    #         # roll_diff, pitch_diff, yaw_diff = tr.euler_from_quaternion(q_lastbaselink_currbaselink)
    #         roll_diff = q_map_currbaselink_euler[0]
    #         pitch_diff = q_map_currbaselink_euler[1]
    #         yaw_diff = q_map_currbaselink_euler[2] - q_map_lastbaselink_euler[2]
    #
    #         self.dyaw_temp += yaw_diff
    #         self.dx_temp += p_lastbaselink_currbaselink[0]
    #         self.dy_temp += p_lastbaselink_currbaselink[1]
    #         #print('dif x',self.dx,'diif y',self.dy,'diff yaw',self.dyaw)


    def predict_next_odometry(self, m):

        delta_x = random.gauss(0, self.dynamics_translation_noise_std_dev) #create gauss
        delta_y = random.gauss(0, self.dynamics_translation_noise_std_dev) #create gauss
        ntheta = random.gauss(0, self.dynamics_orientation_noise_std_dev) #create gauss

        # distance = mt.sqrt(self.dx**2 + self.dy**2)

        if abs(self.dyaw) < 0.1:
            var = self.dx * mt.cos(self.particles[m].theta + self.angx) + self.dy * mt.cos(self.particles[m].theta + self.ang)
            var2 = self.dy*mt.sin(self.particles[m].theta + self.ang) + self.dx*mt.sin(self.particles[m].theta + self.angx)
            flag = 1
        else:
            var2 = 1
            var = 1
            flag = 0

        #print('theta',self.particles[m].theta)
        self.particles[m].pos[1] += (var2*flag + var2*delta_y)/self.map_resolution
        self.particles[m].pos[0] += (var*flag + var*delta_x)/self.map_resolution
        self.particles[m].theta += self.dyaw + ntheta * self.dyaw

        #if abs(self.dyaw) > 0.01:

        #print('VAr x -',self.dpitch,'Var do roll',self.droll)
        #print('The particle',2,'is in (',self.particles[2].pos[0]*self.map_resolution,',',self.particles[2].pos[1]*self.map_resolution,')')
        self.odometry_correct(m) # check if particle is in map
        self.particle_pos_file.write(repr(m)+ ' ' + repr(self.particles[m].pos) + ' ' + repr(self.particles[m].theta)+'\n')
        return [[self.particles[m].pos[0],self.particles[m].pos[1]],self.particles[m].theta]

    def scan_analysis(self, msg):
        if self.first_time == True: # only the first time
            max_angle_sensor = msg.angle_max
            min_angle_sensor = msg.angle_min
            angle_inc_sensor = msg.angle_increment
            self.max_dist = msg.range_max
            self.angle_vect_make(max_angle_sensor, min_angle_sensor, angle_inc_sensor) # create angle vector
            self.first_time = False
        self.ranges_temp = msg.ranges # save ranges

    def get_map(self, msg):
        self.x = msg.info.width
        self.y = msg.info.height
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        data = msg.data
        self.map_resolve_size(data)
        self.map_resolution = msg.info.resolution

    def get_alt(self,msg):
        self.alti = msg.bottom_clearance

    def get_clock(self,msg):
        if self.first_time == True:
            self.time = msg.clock.secs + msg.clock.nsecs * 10**(-9)
        else:
            self.last_time = self.time
            self.time = msg.clock.secs + msg.clock.nsecs * 10**(-9)
            self.dif_t += self.time - self.last_time

    def get_IMU(self,msg):
        self.opt_velocity = msg

    def get_altitude(self,msg):
        self.altitude = msg.ranges[0]

    def velocity_motion_model(self,msg):
        #self.twist_msg = msg.twist.twist

        dif_time = self.dif_t
        self.dif_time = self.dif_t
        self.dif_t = 0

        self.last_velocity = self.curr_velocity
        #self.new_velocity = self.imu_velocity
        self.curr_velocity = msg.twist.twist
        self.last_orientation = self.curr_orientation
        self.curr_orientation = msg.pose.pose

        #print('TIME-',dif_time)

        #self.last_x = self.curr_x
        #self.last_y = self.curr_y


        if  self.last_orientation and self.opt_velocity:

            q_map_lastbaselink = np.array([self.curr_orientation.orientation.x,
                                            self.curr_orientation.orientation.y,
                                            self.curr_orientation.orientation.z,
                                            self.curr_orientation.orientation.w])
            q_map_lastbaselink1 = np.array([self.last_orientation.orientation.x,
                                            self.last_orientation.orientation.y,
                                            self.last_orientation.orientation.z,
                                            self.last_orientation.orientation.w])

            q_map_lastbaselink_euler = euler_from_quaternion(q_map_lastbaselink)
            q_map_lastbaselink_euler1 = euler_from_quaternion(q_map_lastbaselink1)
            self.dyaw_temp += q_map_lastbaselink_euler[2] - q_map_lastbaselink_euler1[2]
            self.droll = q_map_lastbaselink_euler[0] - q_map_lastbaselink_euler1[0]
            self.dpitch = q_map_lastbaselink_euler[1] - q_map_lastbaselink_euler1[1]
            self.yaw = q_map_lastbaselink_euler[2]
            #print('Pitch - ', q_map_lastbaselink_euler[1],'Roll - ', q_map_lastbaselink_euler[0])
            #q_map_lastbaselink = np.array([self.last_velocity.orientation.x,
            #                                self.last_velocity.orientation.y,
            #                                self.last_velocity.orientation.z,
            #                                self.last_velocity.orientation.w])


            #q_map_currbaselink = np.array([self.new_velocity.orientation.x,
            #                                self.new_velocity.orientation.y,
            #                                self.new_velocity.orientation.z,
            #                                self.new_velocity.orientation.w])

            #q_map_lastbaselink_euler = euler_from_quaternion(q_map_lastbaselink)
            #q_map_currbaselink_euler = euler_from_quaternion(q_map_currbaselink)
            #yaw_diff = q_map_currbaselink_euler[2] - q_map_lastbaselink_euler[2]
            #print('Rita-',self.imu_velocity.linear_acceleration.x*(dif_time ** 2))
            #yaw_diff = self.curr_velocity.angular.z * dif_time
            #print('Valor-',yaw_diff)
            #new_x = self.curr_x - (((self.curr_velocity.linear_acceleration.x )*dif_time)/(self.curr_velocity.angular_velocity.x ))* mt.sin(yaw_diff) + (((self.curr_velocity.linear_acceleration.x )*dif_time)/(self.curr_velocity.angular_velocity.x ))* mt.sin(yaw_diff + dif_time * (self.curr_velocity.angular_velocity.x ))
            #new_y = self.curr_y + (((self.curr_velocity.linear_acceleration.y )*dif_time)/(self.curr_velocity.angular_velocity.y ))* mt.cos(yaw_diff) - (((self.curr_velocity.linear_acceleration.y )*dif_time)/(self.curr_velocity.angular_velocity.y ))* mt.cos(yaw_diff + dif_time * (self.curr_velocity.angular_velocity.y ))
            #new_x = (self.curr_velocity.linear.x )*dif_time - self.imu_velocity.linear_acceleration.x*(dif_time ** 2)/2
            #new_y = (self.curr_velocity.linear.y )*dif_time - self.imu_velocity.linear_acceleration.y*(dif_time ** 2)/2
            #self.curr_x = new_x
            #self.curr_y = new_y
            #self.velocity_x += -self.imu_velocity.linear_acceleration.x*(dif_time)
            #self.velocity_y += -self.imu_velocity.linear_acceleration.y*(dif_time)
            #self.dyaw_temp += yaw_diff


            self.forward_x += self.opt_velocity.integrated_y
            self.side_y += self.opt_velocity.integrated_x
            #self.roll +=  self.opt_velocity.integrated_xgyro
            self.pitch =  q_map_lastbaselink_euler1[1]
            #if self.droll < 0.1:
            self.dx_temp += self.curr_velocity.linear.x *dif_time
            #if self.dpitch < 0.1:
            self.dy_temp += self.curr_velocity.linear.y *dif_time

class MCL(object):
    def __init__(self,num_particles):

        # Init node

        rospy.init_node('Monte_Carlo')

        # Errors of associated devices

        dynamics_translation_noise_std_dev   = 0.05
        dynamics_orientation_noise_std_dev   = 0.03
        beam_range_measurement_noise_std_dev = 0.3

        # Get the Particle Filter running

        self.pf = Particle_filter(num_particles,
                                 dynamics_translation_noise_std_dev,
                                 dynamics_orientation_noise_std_dev,
                                 beam_range_measurement_noise_std_dev)

        self.pf.error_file_position = open("error_file_position.txt", "w")
        self.pf.error_file_orientation = open("error_file_orientation.txt", "w")
        self.pf.occupancy_file = open("occupancy_grid.txt", "w")
        self.pf.particle_pos_file = open("particle_position.txt", "w")
        self.pf.ground_truth_file = open("ground_truth_position.txt", "w")
        self.pf.prediction_file = open("prediction_file.txt", "w")

        # Get MAP

        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.gt_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.gt_callback) #/ground_truth/state
        self.pf.Particle_init()
        rospy.Subscriber('/clock', Clock, self.clock_callback)
        self.pf.Initar_init()
        self.gt_yaw = 0
        self.gt_x = 0
        self.gt_y = 0

        self.particles_pub = rospy.Publisher('/typhoon/particle_filter/particles', MarkerArray, queue_size=1)
        # Subscribe to topics

        rospy.Subscriber('/mavros/altitude', Altitude , self.get_altitude) #/ground_truth/state
        #rospy.Subscriber('/mavros/imu/data', Imu, self.odom_callback) #/ground_truth/state
        rospy.Subscriber('/mavros/px4flow/raw/optical_flow_rad', OpticalFlowRad, self.odom_callback) #/ground_truth/state
        rospy.Subscriber('/spur/laser/scan2', LaserScan, self.scan_callback2)
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.twist_callback)
        rospy.Subscriber('/spur/laser/scan', LaserScan, self.scan_callback)



    def scan_callback(self,msg):
        # self.publish_laser_pts(msg)
        self.pf.scan_analysis(msg)

    def scan_callback2(self,msg):
        self.pf.get_altitude(msg)

    def clock_callback(self,msg):
        self.pf.get_clock(msg)

    def gt_callback(self,msg):
        self.pf.get_ground_truth(msg)
        #self.gt_sub.unregister()

    def map_callback(self, msg):
        self.pf.get_map(msg)
        self.map_sub.unregister()

    def get_altitude(self, msg):
        self.pf.get_alt(msg)

    def twist_callback(self,msg):
        self.pf.velocity_motion_model(msg)

    def odom_callback(self, msg):
        #self.pf.odom_processing(msg)
        self.pf.get_IMU(msg)

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
        #print('This is Particle x-',particle.pos[0],'This is Particle y-',particle.pos[1])

        gx = particle.pos[0] * self.pf.map_resolution + self.pf.origin_x
        gy = particle.pos[1] * self.pf.map_resolution + self.pf.origin_y
        quat = Quaternion(*quaternion_from_euler(0,0,particle.theta))
        msg.points.append(Point(gx, gy,0))
        msg.points.append(Point(gx + self.pf.map_resolution*vx, gy + self.pf.map_resolution*vy, 0))
        #msg.pose.orientation = quat

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
            # self.publish_ground_truth()
            self.publish_particle_markers()
            self.pf.Resample_particles()

            rate.sleep()
        self.pf.error_file_position.close()
        self.pf.error_file_orientation.close()
        self.pf.occupancy_file.close()
        self.pf.particle_pos_file.close()
        self.pf.ground_truth_file.close()
        self.pf.prediction_file.close()


if __name__ == '__main__':

    numero_particulas = 200

    Monte_carlo = MCL(numero_particulas)

    Monte_carlo.run()

    #rospy.spin()
