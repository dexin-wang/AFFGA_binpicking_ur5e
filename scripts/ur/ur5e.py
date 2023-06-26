#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#* author: wang dexin
#* unit: Shandong University
#* date: 2023/5/8
#* email: dexinwang@mail.sdu.edu.cn

from geometry_msgs.msg import Pose
import geometry_msgs.msg as geometry_msgs
from .ur5e_cartesian_controler import TrajectoryClient

class UR5e:
    def __init__(self):
        self.client = TrajectoryClient()
        
    def move(self, pose:Pose):
        """将机械手末端移动至输入的位姿
        Pose(geometry_msgs.Vector3(-0.4, 0, 0.1), geometry_msgs.Quaternion(0, 1, 0, 0))
        """
        self.client.send_cartesian_trajectory(pose)
    