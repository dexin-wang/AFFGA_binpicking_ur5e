from tkinter import NO
import cv2
import math
import time
import open3d as o3d
import numpy as np
import scipy.io as scio
from utils.rigid_transformations import RigidTransform
import utils.tool as tool

HEIGHT = 480
WIDTH = 640


class Camera:
    def __init__(self, InMatrix=None, pos=None, rpy=None, transMat=None):
        """
        初始化相机参数，计算相机内参
        假定相机x轴向右，y向下，z向前
        
        pos: 相机在世界坐标系中的位置 [x,y,z]
        rpy: 相机在世界坐标系中的旋转欧拉角 [r,p,y] 
        transMat: 转换矩阵 ndarray shape=(4,4)
        """
        # 1 计算内参矩阵
        if InMatrix is not None:
            self.InMatrix = InMatrix
            self.fxy = InMatrix[0, 0]
        else:
            self.fov = 60   # 垂直视场
            self.fxy = (HEIGHT / 2) / math.tan(tool.angle_TO_radians(self.fov/2))    # 计算 fx 和 fy
            self.InMatrix = np.array([[self.fxy, 0, WIDTH/2 - 0.5], [0, self.fxy, HEIGHT/2 - 0.5], [0, 0, 1]], dtype=np.float64)    # 相机内参

        # 2 世界坐标系->相机坐标系的转换矩阵 4*4
        if transMat is not None:
            self._transMat = transMat
        elif pos is not None and rpy is not None:
            self._transMat = tool.getTransformMat(pos, tool.eulerAnglesToRotationMatrix(rpy))
        else:
            pos = np.array([0, 0, 0])
            rpy = np.array([0, 0, 0])
            self._transMat = tool.getTransformMat(pos, tool.eulerAnglesToRotationMatrix(rpy))
        self._rotation = self._transMat[:3, :3]
        self._translation = self._transMat[:3, 3]

    @property
    def transMat(self):
        ''' 相机坐标系相对于世界坐标系的转换矩阵 '''
        return self._transMat
    
    @property
    def rotation(self):
        ''' 相机坐标系相对于世界坐标系的旋转矩阵 '''
        return self._rotation
    
    @property
    def translation(self):
        ''' 相机坐标系相对于世界坐标系的平移向量 '''
        return self._translation

    @property
    def IntrinsicMatrix(self):
        ''' 相机内参矩阵 '''
        return self.InMatrix

    @property
    def fx(self):
        """ 相机fx """
        return self.InMatrix[0, 0]

    @property
    def fy(self):
        """ 相机fy """
        return self.InMatrix[1, 1]

    @property
    def cx(self):
        """ 相机cx """
        return self.InMatrix[0, 2]
    
    @property
    def cy(self):
        """ 相机cy """
        return self.InMatrix[1, 2]
    
    @property
    def rigidTransform(self):
        """
        返回从world到camera的rigidTransform
        """
        return RigidTransform(rotation=self._rotation , translation=self._translation, 
                              from_frame='world', to_frame='camera')


    def img2camera(self, pt, dep):
        """
        获取像素点pt在相机坐标系中的坐标
        pt: [x, y]
        dep: 深度值

        return: [x, y, z]
        """
        pt_in_img = np.array([[pt[0]], [pt[1]], [1]], dtype=np.float)
        ret = np.matmul(np.linalg.inv(self.InMatrix), pt_in_img) * dep
        return ret.reshape((3,))
    
    def camera2img(self, coord):
        """
        将相机坐标系中的点转换至图像
        coord: [x, y, z]

        return: [col, row]
        """
        z = coord[2]
        coord = np.array(coord).reshape((3, 1))
        rc = (np.matmul(self.InMatrix, coord) / z).reshape((3,))

        return list(rc)[:-1]

    def length_TO_pixels(self, l, dep):
        """
        与相机距离为dep的平面上 有条线，长l，获取这条线在图像中的像素长度
        l: m
        dep: m
        """
        return l * self.fxy / dep

    def pixels_TO_length(self, pixs, dep):
        """
        与相机距离为dep的图像上，有条线，像素长度为pixs, 获取这条线的真实长度
        pixs: 像素长度
        dep: m
        """
        return pixs * dep / self.fxy

    def camera2world(self, coord:np.ndarray, rotM:np.ndarray=None):
        """
        将 相机坐标系中的点和位姿 转换到 世界坐标系中
        corrd: 平移（坐标点） ndarray shape=(3,)
        rotM: 旋转矩阵 ndarray shape=(3,3)

        return: 
            corrd_w: 平移（坐标点） ndarray shape=(3,)
            rotM_w: 旋转矩阵 ndarray shape=(3,3)
        """
        coord = np.r_[np.array(coord), np.array([1,])]
        if rotM is None:
            coord_w = np.matmul(self._transMat, coord.reshape((4, 1))).reshape((4,))
            return coord_w[:-1]
        else:
            # 将坐标和旋转矩阵合并为转换矩阵
            TransM = np.zeros((4, 4), dtype=float)
            TransM[:3, :3] = rotM
            TransM[:, 3] = coord
            
            # 转换到世界坐标系
            ret = np.matmul(self._transMat, TransM)
            corrd_w = ret[:3, 3].reshape((3,))
            rotM_w = ret[:3, :3]
            return corrd_w, rotM_w
    
    def camera2worlds(self, coords):
        """
        获取相机坐标系中的点在世界坐标系中的坐标
        与camera2world不同的是，该函数的输入是一组点
        coords: np.ndarray([[x, y, z], ...])
                shape = (n, 3)

        return: 与输入相同
        """
        ones = np.ones_like((coords.shape[0]))
        coords = np.insert(coords, 3, ones, axis=1).T # (4, n)
        coords_new = np.matmul(self._transMat, coords).T
        return coords_new[:, :-1]
    
    def world2camera(self, coord):
        """
        获取世界坐标系中的点在相机坐标系中的坐标
        corrd: [x, y, z]

        return: [x, y, z]
        """
        coord.append(1.)
        coord = np.array(coord).reshape((4, 1))
        coord_new = np.matmul(np.linalg.inv(self._transMat), coord).reshape((4,))
        return list(coord_new)[:-1]

    def world2img(self, coord):
        """
        获取世界坐标系中的点在图像中的坐标
        corrd: [x, y, z]

        return: [row, col]
        """
        # 转到相机坐标系
        coord = self.world2camera(coord)
        # 转到图像
        pt = self.camera2img(coord) # [x, y]
        return [int(pt[1]), int(pt[0])]
    
    def img2world(self, pt, dep):
        """
        获取像素点的世界坐标
        pt: [x, y]
        dep: 深度值 m
        return: [x, y, z]
        """
        coordInCamera = self.img2camera(pt, dep)
        return self.camera2world(coordInCamera)

    def create_point_cloud(self, im_rgb, im_dep, workspace_mask, organized=True):
        """ Generate point cloud using depth image only.

            Input:
                im_rgb: [numpy.ndarray, (H,W,3), numpy.float32]
                    rgb image
                im_dep: [numpy.ndarray, (H,W), numpy.float32]
                    depth image 单位m
                organized: bool
                    whether to keep the cloud in image shape (H,W,3)

            Output:
                cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                    generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
        """
        # 生成点云
        xmap = np.arange(im_rgb.shape[1])
        ymap = np.arange(im_rgb.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = im_dep
        points_x = (xmap - self.cx) * points_z / self.fx
        points_y = (ymap - self.cy) * points_z / self.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        # print('cloud.shape', cloud.shape)
        if not organized:
            cloud = cloud.reshape([-1, 3])
        
        #cloud: (n, 3)
        # (n, 3) * (3, 3) = (n, 3)
        # rat = np.array([
        #     [1, 0, 0], [0, -1, 0], [0, 0, -1]
        # ])
        # cloud = np.matmul(cloud, rat)

        cloud = cloud[workspace_mask]
        im_rgb = im_rgb[workspace_mask]

        # 转成o3d格式
        pcloud = o3d.geometry.PointCloud()
        pcloud.points = o3d.utility.Vector3dVector(cloud.astype(np.float32))
        pcloud.colors = o3d.utility.Vector3dVector(im_rgb.astype(np.float32))


        return pcloud


# if __name__ == '__main__':
#     camera = Camera(pos=[0, 0, 0.6], rpy=[np.pi, 0, 0])
#     print(camera.InMatrix)