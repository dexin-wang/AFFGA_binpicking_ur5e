#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String


class Robotiq140:
    def __init__(self, pub):
        self.pub = pub
        self.robotiq_cmd = String()

        rospy.loginfo('Init robotiq')
        self.robotiq_cmd.data = 'r' # reset
        self.pub.publish(self.robotiq_cmd)
        rospy.sleep(1)
        self.robotiq_cmd.data = 'a' # activate
        self.pub.publish(self.robotiq_cmd)
        rospy.sleep(3)
        self.move(0.06) # 运动到6cm
        rospy.sleep(1)
        rospy.loginfo('Init robotiq done.')
    
    def move(self, width):
        """
        """
        cmd = int(-14.*width*100.+211.)
        self.robotiq_cmd.data = str(cmd)
        self.pub.publish(self.robotiq_cmd)
        rospy.sleep(1)