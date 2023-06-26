#!/home/cvpr/anaconda3/envs/ros_py37/bin/python3.7

#* author: wang dexin
#* unit: Shandong University
#* date: 2023/5/8
#* email: dexinwang@mail.sdu.edu.cn

"""
抓取位姿检测节点
"""

import rospy
import time
import os
import math
import numpy as np
from std_msgs.msg import Int8
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
# from sensor_msgs.msg import Image
from grasp_binpicking.msg import graspPose
from graspnetAPI import GraspGroup
import geometry_msgs.msg as geometry_msgs
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
sys.path.append('/home/cvpr/catkin_cv_bridge/devel/lib/python3/dist-packages')
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import open3d as o3d
from grasp_methods.affga_net.sgdn.sgdn import SGDN
import grasp_methods.affga_net.sgdn.tool as affga_tool
import utils.tool as utool
from utils.camera import Camera
from utils.grasp import GraspPose, GraspPoses


class GraspPolicy:
    def __init__(self) -> None:
        # ROS节点初始化
        rospy.init_node('grasp_detection')
        rospy.Subscriber("/grasp_start", Int8, self.runGraspDetection)
        self.grasppose_pub = rospy.Publisher('/grasp_pose', graspPose, queue_size=1)
        camera_k = np.array(
            [[613.4265747070312, 0.0, 329.3377990722656], 
            [0.0, 613.5404663085938, 243.68722534179688], 
            [0.0, 0.0, 1.0]])
        self.camera = Camera(camera_k)
    
    def drawPCL(self, im_rgb, im_dep):
        """
        绘制点云
        """
        # 根据input_size生成mask
        workspace_mask = np.zeros(im_rgb.shape[:2], dtype=bool)
        x1 = int((im_rgb.shape[1] - self.input_size) / 2)
        y1 = int((im_rgb.shape[0] - self.input_size) / 2)
        workspace_mask[y1:y1+self.input_size, x1:x1+self.input_size] = 1

        im_rgb = np.array(im_rgb[:, :, ::-1], dtype=np.float32) / 255.0
        pcl = self.camera.create_point_cloud(im_rgb, im_dep, workspace_mask)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='pcd', width=800, height=600)
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters('E:/research/grasp_detection/sim_grasp/sim_dataset_31_qian/scripts/grasp_experiments/view.json')
        vis.add_geometry(pcl)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.run()
        vis.destroy_window()

    def draw6DOFGraspPoses(self, centers, rotate_mats, widths, depths):
        """
        在点云中绘制多个6DOF抓取位姿
        """
        # 根据input_size生成mask
        workspace_mask = np.zeros(self.input_rgb.shape[:2], dtype=bool)
        x1 = int((self.input_rgb.shape[1] - self.input_size) / 2)
        y1 = int((self.input_rgb.shape[0] - self.input_size) / 2)
        workspace_mask[y1:y1+self.input_size, x1:x1+self.input_size] = 1

        im_rgb = np.array(self.input_rgb[:, :, ::-1], dtype=np.float32) / 255.0
        pcl = self.camera.create_point_cloud(im_rgb, self.input_dep, workspace_mask)

        gg_arrays = []
        for i in range(len(centers)):
            # 调整旋转
            offset = utool.eulerAnglesToRotationMatrix([-math.pi/2, -math.pi/2, 0])
            rotMat = np.matmul(rotate_mats[i], offset)
            gg_array1 = np.array([1., widths[i], 0, depths[i]])
            gg_arrays.append(np.concatenate([gg_array1, rotMat.reshape((9,)), centers[i], np.zeros((1,))]))

        gg = GraspGroup(np.array(gg_arrays))
        grippers = gg.to_open3d_geometry_list()
        
        vis = o3d.visualization.Visualizer()    
        vis.create_window(window_name='pcd', width=800, height=600)
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters('/home/cvpr/ur5e_ws/src/view.json')
        vis.add_geometry(pcl)
        for i in range(len(grippers)):
            vis.add_geometry(grippers[i])
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.run()
        # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters('/home/cvpr/ur5e_ws/src/view.json', param)
        vis.destroy_window()


class AFFGANetGraspPolicy(GraspPolicy): 
    def __init__(self, pretrained_model_path):
        super(AFFGANetGraspPolicy, self).__init__()
        # 初始化网络
        # 配置超参数
        self.use_rgb = False  # 是否使用rgb
        self.use_dep = True   # 是否使用深度图
        self.input_size = 384
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print('device = ', device)
        # 模型路径
        model_dir = os.path.join(
            pretrained_model_path,
            'affga_' + 'rgb'*self.use_rgb + 'd'*self.use_dep)
        model_path = os.path.join(model_dir, os.listdir(model_dir)[0])

        # 初始化抓取检测器
        input_channels = int(self.use_rgb)*3+int(self.use_dep)
        self.grasp_model = SGDN('affga', input_channels, model_path, device)

        # 订阅器
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.getAlignedDepth)  # realsense的depth图像
        rospy.Subscriber("/camera/color/image_raw", Image, self.getRGB)  # realsense的RGB图像
        self.wait_alignedDepth = False
        self.wait_rgb = False
        self.img_alignedDepth = None
        self.imgs_alignedDepth = []
        self.img_rgb = None
        self.bridge = CvBridge()

    def getAlignedDepth(self, data):
        """获取与彩色图像对齐的深度图像
        连续5张求均值
        """
        if self.wait_alignedDepth:
            self.imgs_alignedDepth.append(self.bridge.imgmsg_to_cv2(data, "16UC1"))
            # self.img_alignedDepth = self.bridge.imgmsg_to_cv2(data, "16UC1")  # (480, 640)
            # cv2.imwrite('/home/cvpr/wdx/depth.png', img_dep)
            if len(self.imgs_alignedDepth) == 5:
                self.wait_alignedDepth = False
    
    def getRGB(self, data):
        """获取彩色图像"""
        if self.wait_rgb:
            self.img_rgb = self.bridge.imgmsg_to_cv2(data, "bgr8")  # (480, 640)
            # cv2.imwrite('/home/cvpr/wdx/depth.png', img_dep)
            self.wait_rgb = False

    def getInputDatas(self):
        """获取输入网络的观测数据"""
        self.wait_rgb = True
        self.wait_alignedDepth = True

        while self.wait_rgb:
            time.sleep(0.1)
        img_rgb = self.img_rgb
        
        while self.wait_alignedDepth:
            time.sleep(0.1)
        
        ims_d_inpaint = []
        for im in self.imgs_alignedDepth:
            ims_d_inpaint.append(affga_tool.inpaint(im))
        ims_d_inpaint = np.array(ims_d_inpaint)
        img_dep = np.mean(ims_d_inpaint, axis=0)
        # img_dep = self.img_alignedDepth   # 深度图像 单位mm
        # img_dep = affga_tool.inpaint(img_dep)  # 修复
        img_dep = img_dep.astype(np.float32) / 1000.0     # 单位m
        self.imgs_alignedDepth = [] 
        return img_rgb, img_dep

    def runGraspDetection(self, data):
        gp = self.get6DOFGraspPose()
        grasp_pose = graspPose()
        grasp_pose.pose = geometry_msgs.Pose(geometry_msgs.Vector3(gp.center[0], gp.center[1], gp.center[2]+0.01), geometry_msgs.Quaternion(*tuple(gp.quaternion)))
        # grasp_pose.pose = geometry_msgs.Pose(geometry_msgs.Vector3(*tuple(gp.center)), geometry_msgs.Quaternion(*tuple(gp.quaternion)))
        grasp_pose.width = gp.width
        self.grasppose_pub.publish(grasp_pose)
        
    def get6DOFGraspPose(self):
        """获取置信度最高的抓取位姿
        return: GraspPose
        """
        self.input_rgb, self.input_dep = self.getInputDatas()
        grasps = []
        thresh = 0.7
        while len(grasps) == 0:
            grasps = self.grasp_model.predict(
                self.input_rgb, 
                self.input_dep, 
                self.use_rgb, 
                self.use_dep, 
                scale=1.,    #! 0.6/0.6 相机距离工作平面0.6米
                input_size=self.input_size, 
                mode='peak',
                thresh=thresh,
            )
            thresh -= 0.1

        for i in range(len(grasps)):
            row, col, angle, width, conf = tuple(grasps[i]) # width: 单位m
            # width *= 0.8
            row, col = int(row), int(col)
            # 计算 抓取深度
            finger_l1 = 0.03
            finger_l2 = 0.01
            grasp_depth = affga_tool.getGraspDepth(self.input_dep, row, col, angle, 
                                        affga_tool.length_TO_pixels(width, self.input_dep[row, col]), 
                                        affga_tool.length_TO_pixels(finger_l1, self.input_dep[row, col]), 
                                        affga_tool.length_TO_pixels(finger_l2, self.input_dep[row, col]))
            if grasp_depth > 0 or i == len(grasps)-1:
                print('grasp_depth =', grasp_depth)
                # 记录抓取位姿
                grasp_pos_face = self.camera.img2camera([col, row], self.input_dep[row, col])   # 物体表面的抓取点
                grasp_rotMat = utool.eulerAnglesToRotationMatrix([0, 0, -angle+np.pi])   # 旋转矩阵
                # 可视化
                self.draw6DOFGraspPoses([grasp_pos_face,], [grasp_rotMat,], [width,], [grasp_depth,])
                gp = GraspPose()
                gp.from_object_pos(grasp_pos_face, grasp_rotMat, width, grasp_depth, frame='camera')  # 相机坐标系下
                return gp

    def get6DOFGraspPoses(self):
        """获取前50个抓取位姿
        return: GraspPoses
        """
        self.input_rgb, self.input_dep = self.getInputDatas()
        grasps = []
        thresh = 0.7
        while len(grasps) < 50:
            grasps = self.grasp_model.predict(
                self.input_rgb, 
                self.input_dep, 
                self.use_rgb, 
                self.use_dep, 
                scale=1.,    #! 0.6/0.6 相机距离工作平面0.6米
                input_size=self.input_size, 
                mode='peak',
                thresh=thresh
            )
            thresh -= 0.1
        # grasps = [[200, 200, 0, 0.01, 1.0]]   # 测试用
        
        gps = GraspPoses()
        grasp_pos_faces = []
        grasp_rotMats = []
        widths = []
        grasp_depths = []
        for grasp in grasps[:10]:
            row, col, angle, width, conf = tuple(grasp) # width: 单位m
            # width *= 0.8
            row, col = int(row), int(col)
            # 计算 抓取深度
            finger_l1 = 0.03
            finger_l2 = 0.01
            grasp_depth = affga_tool.getGraspDepth(self.input_dep, row, col, angle, 
                                        affga_tool.length_TO_pixels(width, self.input_dep[row, col]), 
                                        affga_tool.length_TO_pixels(finger_l1, self.input_dep[row, col]), 
                                        affga_tool.length_TO_pixels(finger_l2, self.input_dep[row, col]))
            
            # 记录抓取位姿
            grasp_pos_face = self.camera.img2camera([col, row], self.input_dep[row, col])   # 物体表面的抓取点
            grasp_rotMat = utool.eulerAnglesToRotationMatrix([0, 0, -angle])   # 旋转矩阵
            gp = GraspPose()
            gp.from_object_pos(grasp_pos_face, grasp_rotMat, width, grasp_depth, frame='camera')  # 相机坐标系下
            gps.append(gp)

            grasp_pos_faces.append(grasp_pos_face)
            grasp_rotMats.append(grasp_rotMat)
            widths.append(width)
            grasp_depths.append(grasp_depth)

        # 可视化
        self.draw6DOFGraspPoses(grasp_pos_faces, grasp_rotMats, widths, grasp_depths)
            
        return gps

if __name__ == '__main__':
    try:
        pretrained_model_path = 'scripts/grasp_methods/affga_net/pretrained_model'
        policy = AFFGANetGraspPolicy(pretrained_model_path)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

