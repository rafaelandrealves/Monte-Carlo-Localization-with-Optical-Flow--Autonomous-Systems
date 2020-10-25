#!/usr/bin/env python
import rospy
#from nav_msgs.msg import Odometry


#------- For odometry

#def callback(msg):
#    print (msg.pose.pose)


#rospy.init_node('check_odometry')
# print in console that the node is running
#rospy.loginfo('started check_odometry node !')

#odom_sub = rospy.Subscriber('/mavros/local_position/odom',Odometry,callback)

# ----- For LIDAR

from sensor_msgs.msg import LaserScan


def callback(msg):
    print (msg.ranges)


rospy.init_node('check_LIDAR')
#print in console that the node is running
rospy.loginfo('started LIDAR node !')

Lidar_sub = rospy.Subscriber('/spur/laser/scan',LaserScan,callback)




rospy.spin()