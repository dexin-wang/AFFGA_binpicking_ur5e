# -*- coding: utf-8 -*-
"""
@ Time ： 2020/3/2 11:33
@ Auth ： wangdx
@ File ：test-sgdn.py
@ IDE ：PyCharm
@ Function : sgdn测试类
"""

import cv2
import os
import torch
import math
import glob
from sgdn import SGDN

def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasps(img, grasps, mode='line'):
    """
    绘制grasp
    img:    rgb图像
    grasps: list()	元素是 [row, col, angle, width]
    mode:   line or region
    """
    assert mode in ['line', 'region']

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp

        color_b = 255 / num * i
        color_r = 0
        color_g = -255 / num * i + 255

        if mode == 'line':
            width = width / 2

            angle2 = calcAngle2(angle)
            k = math.tan(angle)

            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            if angle < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)


        else:
            img[row, col] = [color_b, color_g, color_r]
        #

    return img

def drawRect(img, rect):
    """
    绘制矩形
    rect: [x1, y1, x2, y2]
    """

    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)


if __name__ == '__main__':
    # 模型路径
    model = 'ckpt/egad-noise-resize/epoch_0195_iou_0.89_'
    input_path = 'img/input'
    output_path = 'img/output'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # 运行设备
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    # 初始化
    sgdn = SGDN(model, device=device_name)
    depth_files = glob.glob(os.path.join(input_path, '*depth.png'))
    for depth_file in depth_files:

        print('processing ', depth_file)

        img = cv2.imread(depth_file, -1)

        # 抓取检测
        grasps, x1, y1 = sgdn.predict(img, device, mode='peak', thresh=0.4, peak_dist=1, angle_k=18)	# 预测
        print(grasps)

        # 绘制抓取检测结果
        rgb_file = depth_file.replace('depth', 'rgb')
        im_rgb = cv2.imread(rgb_file)
        im_grasp = drawGrasps(im_rgb, grasps, mode='line')  # 绘制预测结果
        rect = [x1, y1, x1 + 400, y1 + 400]
        drawRect(im_grasp, rect)
        # 保存检测结果
        save_file = os.path.join(output_path, os.path.basename(rgb_file))
        cv2.imwrite(save_file, im_grasp)

        # 保存特征图
        # able_featureMap, angle_featureMap, width_featureMap = sgdn.maps(img, device)
        # able_featureMap_file = os.path.join(output_path, 'able' + file)
        # angle_featureMap_file = os.path.join(output_path, 'angle' + file)
        # width_featureMap_file = os.path.join(output_path, 'width' + file)
        # cv2.imwrite(able_featureMap_file, able_featureMap)
        # cv2.imwrite(angle_featureMap_file, angle_featureMap)
        # cv2.imwrite(width_featureMap_file, width_featureMap)


    # print('FPS: ', sgdn.fps())
