#!/home/cvpr/anaconda3/envs/ros_py37/bin/python3.7
# -*- coding: utf-8 -*-

#* author: wang dexin
#* unit: Shandong University
#* date: 2023/5/8
#* email: dexinwang@mail.sdu.edu.cn

import rospy
import numpy as np
import time
from geometry_msgs.msg import Pose
import geometry_msgs.msg as geometry_msgs
from ur.ur5e import UR5e
from robotiq import Robotiq140
from std_msgs.msg import Int8, String
from grasp_binpicking.msg import graspPose
import utils.tool as utool
import tf2_ros


class Policy:
    def __init__(self, arm='ur5e', gripper='robotiq140'):
        """初始化 机械臂 / 机械手 / tf2 """
        # ROS节点初始化
        rospy.init_node('grasp_agent')
        # 初始化参数
        self.grasp_start = False
        self.grasp_pose = None
        self.pregrasp_pose = None
        # 订阅
        rospy.Subscriber("/grasp_start", Int8, self.graspStart, queue_size=1, buff_size=1000)
        rospy.Subscriber("/grasp_pose", graspPose, self.updateGraspPose, queue_size=1, buff_size=1000)
        # 发布
        self.robotiq140_pub = rospy.Publisher('/robotiq/gripper_cmd', String, queue_size=10)

        # 初始化机械臂
        if arm == 'ur5e':
            rospy.loginfo('Init ur5e')
            self.arm = UR5e()
            self.watch_pose = geometry_msgs.Pose(geometry_msgs.Vector3(0, -0.4, 0.6-0.25), geometry_msgs.Quaternion(0, 1, 0, 0))
            self.place_pose = geometry_msgs.Pose(geometry_msgs.Vector3(-0.4, 0, 0.6-0.25), geometry_msgs.Quaternion(0, 1, 0, 0))
            self.move_arm(self.watch_pose)
            rospy.loginfo('Init ur5e done')
        else:
            rospy.logerr('Arm set error!')
        # 初始化机械手
        if gripper == 'robotiq140':
            self.gripper = Robotiq140(self.robotiq140_pub)
        else:
            rospy.logerr('Gripper set error!')
        
        # 初始化tf
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
    
    def graspStart(self, data):
        """开始抓取回调函数"""
        # rospy.loginfo('get grasp_start')
        self.grasp_start = True

    def updateGraspPose(self, data:graspPose):
        """获取抓取位姿
        input: 
            data: 相机坐标系下的抓取位姿

        return:
            base坐标系下的抓取位姿和预抓取位姿
        """
        # rospy.loginfo('get grasp_pose')
        # 转到base坐标系下
        try:
            trans_camera = self.tfBuffer.lookup_transform('base', 'camera_color_optical_frame', rospy.Time(0))    # rospy.Time(0) 获取最近的一次转换
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
        t_base_camera = np.array([
            trans_camera.transform.translation.x, 
            trans_camera.transform.translation.y, 
            trans_camera.transform.translation.z])
        q_base_camera = np.array([
            trans_camera.transform.rotation.x, 
            trans_camera.transform.rotation.y, 
            trans_camera.transform.rotation.z,
            trans_camera.transform.rotation.w])
        T_base_camera = utool.getTransformMat(t_base_camera, utool.quaternion_to_rotation_matrix(q_base_camera))

        t_camera_grasp = np.array([
            data.pose.position.x,
            data.pose.position.y,
            data.pose.position.z])
        q_camera_grasp = np.array([
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w])
        T_camera_grasp = utool.getTransformMat(t_camera_grasp, utool.quaternion_to_rotation_matrix(q_camera_grasp))

        # 计算base坐标系下的抓取位姿
        T_base_grasp = np.matmul(T_base_camera, T_camera_grasp)
        q_base_grasp = utool.rotation_matrix_to_quaternion(T_base_grasp[:3, :3])
        self.grasp_pose = graspPose()
        self.grasp_pose.pose = geometry_msgs.Pose(
            geometry_msgs.Vector3(
            T_base_grasp[0, 3], 
            T_base_grasp[1, 3], 
            max(T_base_grasp[2, 3], -0.01)), 
            geometry_msgs.Quaternion(
                q_base_grasp[0],
                q_base_grasp[1],
                q_base_grasp[2],
                q_base_grasp[3]))
        self.grasp_pose.width = data.width

        # 计算base坐标系下的预抓取位姿
        T_grasp_pregrasp = utool.getTransformMat(np.array([0, 0, -0.1]), np.identity(3))
        T_base_pregrasp = np.matmul(T_base_grasp, T_grasp_pregrasp)
        self.pregrasp_pose = graspPose()
        self.pregrasp_pose.pose = geometry_msgs.Pose(
            geometry_msgs.Vector3(
            T_base_pregrasp[0, 3], 
            T_base_pregrasp[1, 3], 
            max(T_base_pregrasp[2, 3], -0.01)), 
            geometry_msgs.Quaternion(
                q_base_grasp[0],
                q_base_grasp[1],
                q_base_grasp[2],
                q_base_grasp[3]))
        self.pregrasp_pose.width = data.width


    def move_gripper(self, width):
        """移动机械手
        width: 米"""
        self.gripper.move(width)

    def move_arm(self, pose:Pose):
        """移动机械臂"""
        self.arm.move(pose)
    
    def run_grasp(self):
        """执行抓取"""
        rospy.loginfo('Witing /grasp_start')
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            if (not self.grasp_start) or (self.grasp_pose is None):
                rate.sleep()
                continue
            rospy.loginfo('================= Start grasp =================')
            # 张开机械手
            rospy.loginfo('step 1: open gripper')
            self.move_gripper(self.grasp_pose.width)
            # 运动至预抓取位姿
            rospy.loginfo('step 2: move to pre-grasp pose')
            self.move_arm(self.pregrasp_pose.pose)
            # 运动至抓取位姿
            rospy.loginfo('step 3: move to grasp pose')
            self.move_arm(self.grasp_pose.pose)
            # 闭合机械手
            rospy.loginfo('step 4: close gripper')
            self.move_gripper(-0.01)
            # 提升机械臂
            rospy.loginfo('step 5: lift arm')
            self.move_arm(self.watch_pose)
            # 移动至收纳盒上方
            rospy.loginfo('step 6: move to place pose')
            self.move_arm(self.place_pose)
            # 张开机械手
            rospy.loginfo('step 7: open gripper')
            self.move_gripper(self.grasp_pose.width)
            # 移动至watch
            rospy.loginfo('step 8: move to watch pose')
            self.move_arm(self.watch_pose)
            rospy.loginfo('================= End grasp =================')
            self.grasp_start = False
            self.grasp_pose = None


if __name__ == '__main__':
    try:
        policy = Policy()
        # rospy.spin()
        policy.run_grasp()
        # policy.move_gripper(0.02)
    except rospy.ROSInterruptException:
        pass
