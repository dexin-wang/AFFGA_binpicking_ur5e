import numpy as np
import utils.tool as tool
from utils.rigid_transformations import RigidTransform
from utils import transformations

class GraspPose:
    """
    一个抓取位姿

    原点为机械手的末端中心，x轴指向任一个夹爪，y轴垂直手掌，z轴与xy轴正交
    """
    def __init__(self, pos=np.zeros(3), rotMat=np.eye(3), width=0, frame='world'):
        """
        创建抓取位姿

        pos: [ndaray, 3, np.float]
            抓取点坐标
        rotMat: [ndaray, 3*3, np.float]
            旋转矩阵
        width: float
            抓取宽度, 单位m
        depth: float
            抓取深度, 单位m
        frame: str
            抓取位姿所在的坐标系
        """
        self.center = pos
        self.rotate_mat = rotMat
        self.width = width
        self.frame = frame
        self.axis = self._axis

    def from_object_pos(self, pos_obj, rotMat, width, depth, frame='world'):
        """
        从物体表面的点创建抓取位姿
        物体表面的点指从像素点投影到物体表面的点

        pos_obj: [ndaray, 3, np.float]
            位于物体表面的点坐标
        rotMat: [ndaray, 3*3, np.float]
            旋转矩阵
        width: float
            抓取宽度, 单位m
        depth: float
            抓取深度,相对于物体表面, 单位m
        frame: str
            抓取位姿所在的坐标系
        """
        self.position_face = pos_obj
        self.rotate_mat = rotMat
        self.width = width
        self.depth = depth
        self.frame = frame
        self.center = self.getCenter()  # 机械手末端中心
        self.axis = self._axis
    

    def from_endpoints(self, p1, p2, frame='world'):
        """
        从抓取的两个端点创建抓取位姿

        p1: np.ndarray shape=(3,)
        alpha: 沿y轴的旋转角 弧度
        """
        self.p1 = p1
        self.p2 = p2
        self.center = (self.p1 + self.p2) / 2
        self.width = np.linalg.norm(p1 - p2)
        self.frame = frame
        grasp_axis = p2 - p1
        self.axis = grasp_axis / np.linalg.norm(grasp_axis)
        self.rotate_mat = self.unrotated_full_axis

    @property
    def unrotated_full_axis(self):
        """ Rotation matrix from canonical grasp reference frame to object reference frame. 
        X axis points out of the gripper palm along the 0-degree approach direction, 
        Y axis points between the jaws, 
        and the Z axs is orthogonal.
        
        本函数输出的抓取旋转,没有考虑approach_angle (即沿抓取坐标系y轴的旋转)

        保证生成的抓取位姿的z轴在frame坐标系z轴上的投影小于0，即在pybullet环境中，机械手向下抓取。

        Returns
        -------
        :obj:`numpy.ndarray`
            rotation matrix of grasp
        """
        grasp_axis_y = self.axis
        grasp_axis_x = np.array([grasp_axis_y[1], -grasp_axis_y[0], 0]) # 使grasp的x轴位于obj的x-y平面上, 即二指机械手的平面 垂直于 桌面 进行抓取
        if np.linalg.norm(grasp_axis_x) == 0:   # 2范数
            grasp_axis_x = np.array([1,0,0])
        grasp_axis_x = grasp_axis_x / np.linalg.norm(grasp_axis_x)    # 归一化
        grasp_axis_z = np.cross(grasp_axis_x, grasp_axis_y)

        if grasp_axis_z[2] > 0:
            grasp_axis_x = np.array([-grasp_axis_y[1], grasp_axis_y[0], 0]) # 使grasp的x轴位于obj的x-y平面上, 即二指机械手的平面 垂直于 桌面 进行抓取
            if np.linalg.norm(grasp_axis_x) == 0:   # 2范数
                grasp_axis_x = np.array([1,0,0])
            grasp_axis_x = grasp_axis_x / np.linalg.norm(grasp_axis_x)    # 归一化
            grasp_axis_z = np.cross(grasp_axis_x, grasp_axis_y)
        
        R = np.c_[grasp_axis_x, np.c_[grasp_axis_y, grasp_axis_z]]  # 先把每个(3,)向量reshape成(3,1),再沿列方向合并
        return R


    @staticmethod
    def _get_rotation_matrix_y(theta):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.c_[[cos_t, 0, sin_t], np.c_[[0, 1, 0], [-sin_t, 0, cos_t]]]
        return R
    
    @property
    def _pts(self):
        """
        计算抓取的两个端点

        抓取中心点 - 抓取宽度乘以axis        抓取位姿y轴上的
        """
        self.p1 = self.center - (self.width / 2) * self._axis
        self.p1 = self.center + (self.width / 2) * self._axis

    @property
    def zz_axis(self):
        """:obj:`numpy.ndarray` of float: 不考虑平移，to_frame的z轴在from_frame的z轴的投影
        """
        return self.rotate_mat[2,2]

    @property
    def _axis(self):
        """
        抓取轴 / 两端点的差 / grasp坐标系的y轴在frame坐标系中的投影 / 只考虑旋转，grasp坐标系的(0,1,0)在frame坐标系中的坐标
        """
        return self.rotate_mat[:,1]

    def getCenter(self):
        """
        计算机械手末端中心的点
        """
        # 设 g为物体表面的抓取位姿，g1为实际的抓取位姿，c为相机坐标系
        # T_c_g1 = T_c_g * T_g_g1   T_c_g已知
        T_g_g1 = np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0],
            [0, 0, 1, self.depth],
            [0, 0, 0, 1]
        ])
        T_c_g = tool.getTransformMat(self.position_face, self.rotate_mat)
        T_c_g1 = np.matmul(T_c_g, T_g_g1)

        center = T_c_g1[:3, 3].reshape((3,))
        return center

    @property
    def quaternion(self):
        """
        return:
            四元数
        """
        M = np.identity(4)
        M[:3, :3] = self.rotate_mat
        # return transformations.quaternion_from_matrix(self.rotate_mat)
        return transformations.quaternion_from_matrix(M)

    @property
    def rigidTransform(self):
        """
        返回从'frame'到 grasp 的 rigidTransform
        """
        return RigidTransform(rotation=self.rotate_mat, translation=self.center, 
                              from_frame=self.frame, to_frame='grasp')
    
    def transform(self, rigidTransform:RigidTransform):
        """
        将抓取位姿转换到 rigidTransform 的 from_frame 下
        """
        assert rigidTransform.to_frame == self.frame
        T_frame_grasp = RigidTransform(rotation=self.rotate_mat, translation=self.center, from_frame=self.frame, to_frame='grasp')
        T_newframe_grasp = rigidTransform.dot(T_frame_grasp)
        self.rotate_mat = T_newframe_grasp.rotation
        self.center = T_newframe_grasp.translation
        self.frame = T_newframe_grasp.from_frame

    
    def transformPose(self, transMat):
        """
        根据输入的转换矩阵将抓取位姿转换到相应的坐标系下(左乘)
        transMat: [ndarray, (4,4), np.float]
            相机坐标系相对于其他坐标系e的转换矩阵
        """
        # T_e_g = T_e_c * T_c_g
        rotate_mat = self.rotate_mat
        # 转换实际抓取位姿
        T_c_g1 = tool.getTransformMat(self.position, rotate_mat)
        T_e_g1 = np.matmul(transMat, T_c_g1)
        self.position = T_e_g1[:3, 3].reshape((3,))
        self.rotate_mat = T_e_g1[:3, :3]
        # print('self.rotate_mat 1 = ', self.rotate_mat)

        # 转换物体表面抓取位姿
        T_c_g = tool.getTransformMat(self.position_face, rotate_mat)
        T_e_g = np.matmul(transMat, T_c_g)
        self.position_face = T_e_g[:3, 3].reshape((3,))
        self.rotate_mat = T_e_g[:3, :3]
    


class GraspPoses:
    """
    多个抓取位姿
    """
    def __init__(self):
        self.grasp_poses = []

    def append(self, gp:GraspPose):
        self.grasp_poses.append(gp)
        pass
    
    def __len__(self):
        """
        返回抓取位姿的数量
        """
        return len(self.grasp_poses)

    def __getitem__(self, idx):
        """
        返回idx索引的抓取位姿
        """
        return self.grasp_poses[idx]