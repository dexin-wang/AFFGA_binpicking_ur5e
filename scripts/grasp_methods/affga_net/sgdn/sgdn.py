#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@ Time ： 2020/3/2 11:33
@ Auth ： wangdx    
@ File ：test-sgdn.py
@ IDE ：PyCharm
@ Function : sgdn测试类
"""
import sys
import os


import cv2
import torch
import time
import math
from skimage.feature import peak_local_max
import numpy as np
from grasp_methods.affga_net.sgdn.models.ggcnn.common import post_process_output
from grasp_methods.affga_net.sgdn.models.loss import get_pred
from grasp_methods.affga_net.sgdn.models import get_network
# from grasp_methods.sgdn.models.SWIN1.config import getconfig 
# from grasp_methods.affga_net.sgdn.models.SWIN.config import getconfig 
# from grasp_methods.affga_net.sgdn.models.SWIN1.config import getconfig as getconfig1

# from skimage.draw import line
from .utils.img import Image, RGBImage, DepthImage
# from grasp_methods.affga_net.sgdn.models.SWIN.graspnet import GraspNet2
# from grasp_methods.affga_net.sgdn.models.SWIN1.graspnet import GraspNet
def depth2Gray(im_depth):
    """
    将深度图转至三通道8位灰度图
    (h, w, 3)
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max

    ret = (im_depth * k + b).astype(np.uint8)
    return ret

def arg_thresh(array, thresh):
    """
    获取array中大于thresh的二维索引
    array: 二维array
    thresh: float阈值
    :return: array shape=(n, 2)
    """
    res = np.where(array > thresh)
    rows = np.reshape(res[0], (-1, 1))
    cols = np.reshape(res[1], (-1, 1))
    locs = np.hstack((rows, cols))
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0])[i+1:]:
            if array[locs[i, 0], locs[i, 1]] < array[locs[j, 0], locs[j, 1]]:
                locs[[i, j], :] = locs[[j, i], :]

    return locs


class SGDN:
    def __init__(self, net, input_channels, model, device):
        """
        net: 网络架构 'ggcnn2', 'deeplabv3', 'grcnn', 'unet', 'segnet', 'stdc', 'danet'
        input_channels: 输入通道数 1/3/4
        model: 训练好的模型路径
        device: cpu/cuda:0
        """
        print('>> loading SGDN')
        sgdn = get_network(net)
        self.net = sgdn(input_channels=input_channels, angle_cls=18)
        self.device = torch.device(device)
        self.net.load_state_dict(torch.load(model, map_location=self.device), strict=True)
        print('>> load done')

    @staticmethod
    def numpy_to_torch(s):
        """
        numpy转tensor
        """
        if len(s.shape) == 2:
            return torch.from_numpy(s[np.newaxis, np.newaxis, :, :].astype(np.float32))
        elif len(s.shape) == 3:
            return torch.from_numpy(s[np.newaxis, :, :, :].astype(np.float32))
        else:
            raise np.AxisError

    def predict(self, img_rgb, img_dep, use_rgb, use_dep, scale, input_size, mode, thresh=0.5, peak_dist=1, angle_k=18):
        """
        预测抓取模型
        img_rgb: rgb图像 np.array (h, w, 3)
        img_dep: 深度图 np.array (h, w)
        use_rgb: 是否使用RGB图像
        use_dep: 是否使用深度图像
        scale: 输入的图像需要resize,使图像高度位于数据集中相机的高度处
        input_size: 图像送入神经网络时的尺寸，需要在原图上裁剪
        mode: max, peak, all, nms
        thresh: 置信度阈值
        peak_dist: 置信度筛选峰值
        angle_k: 抓取角分类数
        :return:
            pred_grasps: list([row, col, angle, width])  width单位为米
        """
        # 获取输入tensor
        image = Image()
        if use_rgb:
            im_rgb = RGBImage(img_rgb)
            image.rgbimg = im_rgb
        if use_dep:
            im_dep = DepthImage(img_dep)
            image.depthimg = im_dep
        # 先根据相机进行resize,再裁剪中间区域
        image.rescale(scale)
        self.crop_x1, self.crop_y1, _, _ = image.crop(input_size)
        input = self.numpy_to_torch(image.nomalise())    # (n, h, w) / (h, w)

        # 预测
        self.net.eval()
        with torch.no_grad():
            self.able_out, self.angle_out, self.width_out = get_pred(self.net.to(self.device), input.to(self.device))

            able_pred, angle_pred, width_pred = post_process_output(self.able_out, self.angle_out, self.width_out, GRASP_WIDTH_MAX=0.1)

            if mode == 'peak':
                # 置信度峰值 抓取点
                pred_pts = peak_local_max(able_pred, min_distance=peak_dist, threshold_abs=thresh)
            elif mode == 'all':
                # 超过阈值的所有抓取点
                pred_pts = arg_thresh(able_pred, thresh=thresh)
            elif mode == 'max':
                # 置信度最大的点
                loc = np.argmax(able_pred)
                row = loc // able_pred.shape[0]
                col = loc % able_pred.shape[0]
                pred_pts = np.array([[row, col]])
            else:
                raise ValueError

            pred_grasps = []
            for idx in range(pred_pts.shape[0]):
                row, col = pred_pts[idx]
                conf = able_pred[row, col]
                angle = angle_pred[row, col] * 1.0 / angle_k * np.pi  # 预测的抓取角弧度
                width = width_pred[row, col]    # 实际长度 m
                row += self.crop_y1
                col += self.crop_x1
                row = int(row * 1.0 / scale)
                col = int(col * 1.0 / scale)

                pred_grasps.append([row, col, angle, width, conf])

            print('output grasp num: ', len(pred_grasps))
            if len(pred_grasps) == 0:
                return pred_grasps
                
            pred_grasps = np.array(pred_grasps, dtype=np.float)
            # 对pred_grasps排序
            idxs = np.argsort(pred_grasps[:, 4] * -1)
            pred_grasps = pred_grasps[idxs]

            return pred_grasps


    