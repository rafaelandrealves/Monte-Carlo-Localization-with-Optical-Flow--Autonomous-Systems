#!/usr/bin/env python
# -- coding: utf-8 --
import numpy as np
import math as mt
import rospy
from sensor_msgs.msg import LaserScan
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

        self.dpitch = 0
        self.droll = 0
        self.dyaw = 0
        self.dx = 0
        self.dy = 0
        self.static = 1

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
            # if(m==5):
            #     self.particles.append(Particle(m,[422,384],1))
            #     print('map=',self.map[422,384])
            # else:
            self.particles.append(Particle(m,[idxf,idy],1)) # append particle to particle filter
            #print('map=',self.map[idxf,idy])

    def Init_one_particle(self, _m):
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
        self.particles[_m].theta = 0

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

    def Resample_particles(self):
        while self.max_dist == 0: #while not started
            rospy.sleep(1)
        self.ranges = self.ranges_temp #assign to local variable
        newPF = [] # X~
        finPF = [] # X
        new_weight = np.zeros(self.M) #new vector for weight change
        #rospy.sleep(10)
        self.m_to_grid() # convert from meters to grid coordinates
        for m in range(self.M):
            [new_pos,new_theta] = self.predict_next_odometry(m) #predict odom
            new_weight[m] = self.weight_change(m) # predict weight
            self.particles[m].w = new_weight[m] # assign to particle
            #if m == 1:
            newPF.append(Particle(m, new_pos, new_weight[m], _theta = new_theta)) #create new PF
        new_weight = self.normalize_weights(new_weight)
        eff_particles = self.det_eff_part(new_weight)
        #print('A particula 2 esta na posica x:',self.particles[2].pos[0],'e y:',self.particles[2].pos[1],'\n')
        #print(new_weight)
        #print('EFfective particle:',eff_particles)
        b_idx = np.argmax(new_weight) #get best prediction index
        max_weight = max(new_weight) #get best prediction
        #print('The best estimate is given by x = ', self.particles[b_idx].pos[0]*self.map_resolution,' and y = ', self.particles[b_idx].pos[1]*self.map_resolution,' with weight = ', max_weight)
        if eff_particles < 2*(self.M)/3:
            w_v = np.array(new_weight)
            w_v = w_v*self.M
            for m in range(self.M):
                if(max(w_v)<1):
                    mx = np.argmax(w_v)
                    w_v[mx] = 0
                else:
                    mx = np.argmax(w_v) #max probability from matrix
                    w_v[mx] = w_v[mx] - 1
                idx = self.particles[mx].pos[0]
                idy = self.particles[mx].pos[1]
                finPF.append(Particle(m,[idx,idy],1, _theta = self.particles[m].theta))
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

    def m_to_grid(self):
        self.ranges_in_grid = np.zeros((2,self.angle_readings)) #create new ranges vector
        for i in range(self.angle_readings):
            if self.ranges[i] < self.max_dist and self.ranges[i] > self.min_dist: # in range
                self.ranges_in_grid[0,i] = self.ranges[i]/self.map_resolution #convert from meters to grid
                self.ranges_in_grid[1,i] = self.ranges[i]/self.map_resolution #convert from meters to grid
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
        ang_dist_x = mt.cos(self.angle_vector[_i]+self.particles[_m].theta)*self.ranges_in_grid[0,_i] #
        ang_dist_y = mt.sin(self.angle_vector[_i]+self.particles[_m].theta)*self.ranges_in_grid[1,_i] #trigonometry
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

    def weight_change(self, _m):
        if self.static == 0: #verify if it is moving
            return self.particles[_m].w
        wt = 1 # temporary weight
        for i in range(self.angle_readings): # for all laser readings
            if self.ranges_in_grid[0,i] != -1: # check if is valid
                wt = wt + self.compare_dist(_m,i) # change weight
        return wt

    def odometry_correct(self, _m):
        xx = int(self.particles[_m].pos[0]) # pos x from particle
        yy = int(self.particles[_m].pos[1]) # pos y from particle
        if xx >= self.x or xx < 0 or yy >= self.y or yy < 0: #check if it is outside map
            self.Init_one_particle(_m)
            return
        if self.map[xx,yy] != 0: #check if it is in available place
            self.Init_one_particle(_m)

    def odom_processing(self,msg):
        #Save robot Odometry


        # Determine the difference between new and old values
        self.last_robot_odom = self.robot_odom
        self.robot_odom = msg

        if self.last_robot_odom: #if its not the first time

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
            p_lastbaselink_currbaselink = p_map_currbaselink-p_map_lastbaselink
            # R_map_lastbaselink = tr.quaternion_matrix(q_map_lastbaselink)[0:3,0:3]
            #
            # p_lastbaselink_currbaselink = R_map_lastbaselink.transpose().dot(p_map_currbaselink - p_map_lastbaselink)

            q_map_lastbaselink_euler = euler_from_quaternion(q_map_lastbaselink)
            q_map_currbaselink_euler = euler_from_quaternion(q_map_currbaselink)
            # q_lastbaselink_currbaselink = tr.quaternion_multiply(tr.quaternion_inverse(q_map_lastbaselink), q_map_currbaselink)
            #
            # roll_diff, pitch_diff, yaw_diff = tr.euler_from_quaternion(q_lastbaselink_currbaselink)
            roll_diff = q_map_currbaselink_euler[0]
            pitch_diff = q_map_currbaselink_euler[1]
            yaw_diff = q_map_currbaselink_euler[2] - q_map_lastbaselink_euler[2]


            self.droll += roll_diff
            self.dpitch += pitch_diff
            self.dyaw += yaw_diff
            self.dx += p_lastbaselink_currbaselink[0]
            self.dy += p_lastbaselink_currbaselink[1]
            #print('dif x',self.dx,'diif y',self.dy,'diff yaw',self.dyaw)


    def predict_next_odometry(self, m):

        delta_x = random.gauss(0, self.dynamics_translation_noise_std_dev) #create gauss
        delta_y = random.gauss(0, self.dynamics_translation_noise_std_dev) #create gauss
        ntheta = random.gauss(0, self.dynamics_orientation_noise_std_dev) #create gauss

        # distance = mt.sqrt(self.dx**2 + self.dy**2)

        if(abs(self.dpitch) > 0.5 or abs(self.droll) > 0.05): # check if is moving
            self.static = 0
        else:
            self.static = 1

        if (abs(self.dx) > 0.01): #check if was moving
            self.particles[m].pos[0] += (self.dx + delta_x)/self.map_resolution  # * (self.dx)/self.map_resolution
            self.static = 0

        if (abs(self.dy) > 0.01):
            self.particles[m].pos[1] += (self.dy + delta_y)/self.map_resolution  # * (self.dy)/self.map_resolution
            self.static = 0

        if abs(self.dyaw) > 0.01:
            self.particles[m].theta += self.dyaw + ntheta #* self.dyaw
            self.static = 1

        #print('VAr x -',self.dpitch,'Var do roll',self.droll)
        #print('The particle',2,'is in (',self.particles[2].pos[0]*self.map_resolution,',',self.particles[2].pos[1]*self.map_resolution,')')
        self.odometry_correct(m) # check if particle is in map
        return [[self.particles[m].pos[0],self.particles[m].pos[1]],self.particles[m].theta]

    def scan_analysis(self, msg):
        if self.first_time == True: # only the first time
            max_angle_sensor = msg.angle_max
            min_angle_sensor = msg.angle_min
            angle_inc_sensor = msg.angle_increment
            self.max_dist = msg.range_max
            self.min_dist = msg.range_min
            self.angle_vect_make(max_angle_sensor, min_angle_sensor, angle_inc_sensor) # create angle vector
            self.first_time = False
        self.ranges_temp = msg.ranges # save ranges

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

        dynamics_translation_noise_std_dev   = 0.0015
        dynamics_orientation_noise_std_dev   = 0.03
        beam_range_measurement_noise_std_dev = 0.3

        # Get the Particle Filter running

        self.pf = Particle_filter(num_particles,
                                 dynamics_translation_noise_std_dev,
                                 dynamics_orientation_noise_std_dev,
                                 beam_range_measurement_noise_std_dev)

        # Get MAP

        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

        self.pf.Particle_init()
        self.gt_yaw = 0
        self.gt_x = 0
        self.gt_y = 0

        #self.laser_points_marker_pub = rospy.Publisher('/typhoon/debug/laser_points', Marker, queue_size=1)
        self.particles_pub = rospy.Publisher('/typhoon/particle_filter/particles', MarkerArray, queue_size=1)
        # self.gt_pub = rospy.Publisher('/typhoon/particle_filter/ground_truth', Marker, queue_size=1)
        # Subscribe to topics
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback) #/ground_truth/state
        # rospy.Subscriber('/ground_truth/state', Odometry, self.gt_processing) #/ground_truth/state
        rospy.Subscriber('/spur/laser/scan', LaserScan, self.scan_callback)



    def scan_callback(self,msg):
        # self.publish_laser_pts(msg)
        self.pf.scan_analysis(msg)


    def map_callback(self, msg):
        self.pf.get_map(msg)
        self.map_sub.unregister()

    def odom_callback(self, msg):
        self.pf.odom_processing(msg)

    # def gt_processing(self,msg):
    #     #Save robot Odometry
    #
    #
    #     # Determine the difference between new and old values
    #     self.robot_odom = msg
    #
    #     p_map_currbaselink = np.array([self.robot_odom.pose.pose.position.x,
    #                                     self.robot_odom.pose.pose.position.y,
    #                                     self.robot_odom.pose.pose.position.z])
    #
    #     q_map_currbaselink = np.array([self.robot_odom.pose.pose.orientation.x,
    #                                     self.robot_odom.pose.pose.orientation.y,
    #                                     self.robot_odom.pose.pose.orientation.z,
    #                                     self.robot_odom.pose.pose.orientation.w])
    #
    #     # Save quaternion units, with axis of rotation
    #     # Does the rotation matrix
    #     p_lastbaselink_currbaselink = p_map_currbaselink
    #     # R_map_lastbaselink = tr.quaternion_matrix(q_map_lastbaselink)[0:3,0:3]
    #     #
    #     # p_lastbaselink_currbaselink = R_map_lastbaselink.transpose().dot(p_map_currbaselink - p_map_lastbaselink)
    #
    #     q_map_currbaselink_euler = euler_from_quaternion(q_map_currbaselink)
    #     # q_lastbaselink_currbaselink = tr.quaternion_multiply(tr.quaternion_inverse(q_map_lastbaselink), q_map_currbaselink)
    #     #
    #     # roll_diff, pitch_diff, yaw_diff = tr.euler_from_quaternion(q_lastbaselink_currbaselink)
    #
    #     yaw_diff = q_map_currbaselink_euler[2]
    #
    #     self.gt_yaw = yaw_diff
    #     self.gt_x = p_lastbaselink_currbaselink[0]
    #     self.gt_y = p_lastbaselink_currbaselink[1]

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
    #
    # def publish_laser_pts(self, msg):
    #     """Publishes the currently received laser scan points from the robot, after we subsampled
    #     them in order to comparse them with the expected laser scan from each particle."""
    #     if self.pf.robot_odom is None:
    #         return
    #
    #     subsampled_ranges, subsampled_angles = self.pf.subsample(msg)
    #
    #
    #     N = len(subsampled_ranges)
    #     x = self.pf.robot_odom.pose.pose.position.x
    #     y = self.pf.robot_odom.pose.pose.position.y
    #     _, _ , yaw_in_map = tr.euler_from_quaternion(np.array([self.pf.robot_odom.pose.pose.orientation.x,
    #                                                            self.pf.robot_odom.pose.pose.orientation.y,
    #                                                            self.pf.robot_odom.pose.pose.orientation.z,
    #                                                            self.pf.robot_odom.pose.pose.orientation.w]))
    #
    #     pts_in_map = [ (x ,
    #                     y ,
    #                     self.pf.map_resolution) for r,theta in zip(subsampled_ranges, subsampled_angles)]
    #
    #
    #     # lpmarker = self.get_2d_laser_points_marker(msg.header.stamp, 'map', pts_in_map, 30000, ColorRGBA(0, 0.0, 0, 1.0))
    #     # self.laser_points_marker_pub.publish(lpmarker)


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
        # if marker_id == 5:
        #     msg.color = ColorRGBA(1.0, 0.0, 0, 1.0)

        #print('This is Particle x-',particle.pos[0],'This is Particle y-',particle.pos[1])

        gx = particle.pos[0] * self.pf.map_resolution - 10
        gy = particle.pos[1] * self.pf.map_resolution - 10
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

    # def get_gt_marker(self, timestamp):
    #     """Returns an rviz marker that visualizes a single particle"""
    #     msg = Marker()
    #     msg.header.stamp = timestamp
    #     msg.header.frame_id = 'map'
    #     msg.ns = 'ground_truth'
    #     msg.id = 0
    #     msg.type = 0  # arrow
    #     msg.action = 0 # add/modify
    #     msg.lifetime = rospy.Duration(1)
    #
    #     yaw_in_map = self.gt_yaw
    #     vx = mt.cos(yaw_in_map)
    #     vy = mt.sin(yaw_in_map)
    #     msg.color = ColorRGBA(1.0, 0, 0, 1.0)
    #
    #     #print('This is Particle x-',particle.pos[0],'This is Particle y-',particle.pos[1])
    #
    #     gx = self.gt_x - 10
    #     gy = self.gt_y - 10
    #     quat = Quaternion(*quaternion_from_euler(0,0,self.gt_yaw))
    #     msg.points.append(Point(gx, gy,0))
    #     msg.points.append(Point(gx + self.pf.map_resolution*vx, gy + self.pf.map_resolution*vy, 0))
    #     msg.pose.orientation = quat
    #
    #     msg.scale.x = 0.3
    #     msg.scale.y = 0.5
    #     msg.scale.z = 1
    #     return msg
    #
    # def publish_ground_truth(self):
    #     ts = rospy.Time.now()
    #     gt = self.get_gt_marker(ts)
    #
    #     self.gt_pub.publish(gt)

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            # self.publish_ground_truth()
            self.publish_particle_markers()
            self.pf.Resample_particles()
            self.pf.dx = 0
            self.pf.dy = 0
            self.pf.dyaw = 0
            self.pf.droll = 0
            self.pf.dpitch = 0

            rate.sleep()


if __name__ == '__main__':

    numero_particulas = 100

    Monte_carlo = MCL(numero_particulas)

    Monte_carlo.run()

    #rospy.spin()
