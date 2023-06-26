import math
import numpy as np
from skimage.draw import line
import cv2


def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img

def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasps(img, grasps, im_dep):
    """
    绘制grasp
    file: img路径
    grasps: list()	元素是 [row, col, angle, width, conf]
    width: 米
    """
    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width, conf = grasp
        row = int(row)
        col = int(col)
        if im_dep is not None:
            depth = im_dep[row, col]
            # 方法1:使用固定值
            # width_pixel = width * 971.66  # 米->像素  888.89     固定深度
            # 方法2:根据抓取点的深度
            # width_pixel = length2pixel(width, depth) / 2
            width_pixel = length_TO_pixels(width, depth) / 2
        else:
            width_pixel = width

        angle2 = calcAngle2(angle)
        k = math.tan(angle)

        if k == 0:
            dx = width_pixel
            dy = 0
        else:
            dx = k / abs(k) * width_pixel / pow(k ** 2 + 1, 0.5)
            dy = k * dx

        if angle < math.pi:
            # cv2.arrowedLine(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 3, 8, 0, 0.3)
            cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 3)
        else:
            # cv2.arrowedLine(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 3, 8, 0, 0.3)
            cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 3)

        if angle2 < math.pi:
            cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 3)
        else:
            cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 3)

        color_b = 255 / num * i
        color_r = 0
        color_g = -255 / num * i + 255
        
        # img[row, col] = [color_b, color_g, color_r]
        cv2.circle(img, (col, row), 5, (color_b, color_g, color_r), -1)

    return img

def minDepth(img, row, col, l):
    """
    获取(row, col)周围l*l范围内最小的深度值
    :param img: 单通道深度图
    :param l: 3, 5, 7, 9 ...
    :return: float
    """
    row_t = row - (l - 1) / 2
    row_b = row + (l - 1) / 2
    col_l = col - (l - 1) / 2
    col_r = col + (l - 1) / 2

    min_depth = 10000

    for r in range(row_b + 1)[row_t + 1:]:
        for c in range(col_r + 1)[col_l + 1:]:
            dep = img[int(r), int(c)]
            if 1 < dep < min_depth:
                min_depth = dep
            # print('row = {}, col = {}'.format(r, c))
    return min_depth

def ptsOnRect(pts):
    """
    获取矩形框上五条线上的点
    五条线分别是：四条边缘线，1条对角线
    pts: np.array, shape=(4, 2) (row, col)
    """
    rows1, cols1 = line(int(pts[0, 0]), int(pts[0, 1]), int(pts[1, 0]), int(pts[1, 1]))
    rows2, cols2 = line(int(pts[1, 0]), int(pts[1, 1]), int(pts[2, 0]), int(pts[2, 1]))
    rows3, cols3 = line(int(pts[2, 0]), int(pts[2, 1]), int(pts[3, 0]), int(pts[3, 1]))
    rows4, cols4 = line(int(pts[3, 0]), int(pts[3, 1]), int(pts[0, 0]), int(pts[0, 1]))
    rows5, cols5 = line(int(pts[0, 0]), int(pts[0, 1]), int(pts[2, 0]), int(pts[2, 1]))

    rows = np.concatenate((rows1, rows2, rows3, rows4, rows5), axis=0)
    cols = np.concatenate((cols1, cols2, cols3, cols4, cols5), axis=0)
    return rows, cols

def ptsOnRotateRect(pt1, pt2, w, img_dep_rgb=None):
    """
    绘制矩形
    已知图像中的两个点（x1, y1）和（x2, y2），以这两个点为端点画线段，线段的宽是w。这样就在图像中画了一个矩形。
    pt1: [row, col] 
    w: 单位像素
    img: 绘制矩形的图像, 单通道
    """
    y1, x1 = pt1
    y2, x2 = pt2

    if x2 == x1:
        if y1 > y2:
            angle = math.pi / 2
        else:
            angle = 3 * math.pi / 2
    else:
        tan = (y1 - y2) * 1.0 / (x2 - x1)
        angle = np.arctan(tan)

    points = []
    points.append([y1 - w / 2 * np.cos(angle), x1 - w / 2 * np.sin(angle)])
    points.append([y2 - w / 2 * np.cos(angle), x2 - w / 2 * np.sin(angle)])
    points.append([y2 + w / 2 * np.cos(angle), x2 + w / 2 * np.sin(angle)])
    points.append([y1 + w / 2 * np.cos(angle), x1 + w / 2 * np.sin(angle)])
    points = np.array(points)

    if img_dep_rgb is not None:
        cv2.circle(img_dep_rgb, (int(points[0][1]), int(points[0][0])), 3, (0, 0, 0), -1) 
        cv2.circle(img_dep_rgb, (int(points[1][1]), int(points[1][0])), 3, (0, 0, 0), -1) 
        cv2.circle(img_dep_rgb, (int(points[2][1]), int(points[2][0])), 3, (0, 0, 0), -1) 
        cv2.circle(img_dep_rgb, (int(points[3][1]), int(points[3][0])), 3, (0, 0, 0), -1) 

    # 方案1，比较精确，但耗时
    # rows, cols = polygon(points[:, 0], points[:, 1], (10000, 10000))	# 得到矩形中所有点的行和列

    # 方案2，速度快
    return ptsOnRect(points)	# 得到矩形中所有点的行和列

def collision_detection(pt, dep, angle, depth_map, finger_l1, finger_l2, img_dep_rgb=None):
    """
    碰撞检测
    pt: (row, col)
    angle: 抓取角 弧度
    depth_map: 深度图
    finger_l1 l2: 像素长度

    return:
        True: 无碰撞
        False: 有碰撞
    """
    row, col = pt
    H, W = depth_map.shape

    # 两个点
    row1 = int(row - finger_l2 * math.sin(angle))
    col1 = int(col + finger_l2 * math.cos(angle))

    row1 = max(min(row1, H-1), 0)
    col1 = max(min(col1, W-1), 0)

    if img_dep_rgb is not None:
        cv2.circle(img_dep_rgb, (col1, row1), 3, (0, 0, 0), -1) 
    
    # 在截面图上绘制抓取器矩形
    # 检测截面图的矩形区域内是否有1
    rows, cols = ptsOnRotateRect([row, col], [row1, col1], finger_l1, img_dep_rgb)

    try:
        # print('np.min(depth_map[rows, cols]) = ', np.min(depth_map[rows, cols]))
        # print('dep = ', dep)
        if np.min(depth_map[rows, cols]) > dep:   # 无碰撞
            # print('无碰撞')
            return True
    except:
        return True    # 有碰撞
    
    return False    # 有碰撞

def getGraspDepth(camera_depth, grasp_row, grasp_col, grasp_angle, grasp_width, finger_l1, finger_l2, img_dep_rgb=None):
    """
    根据深度图像及抓取角、抓取宽度，计算最大的无碰撞抓取深度（相对于物体表面的下降深度）
    此时抓取点为深度图像的中心点
    camera_depth: 位于抓取点正上方的相机深度图
    grasp_angle：抓取角 弧度
    grasp_width：抓取宽度 像素
    finger_l1 l2: 抓取器尺寸 像素长度

    return: 抓取深度，相对于相机的深度
    """
    # grasp_row = int(camera_depth.shape[0] / 2)
    # grasp_col = int(camera_depth.shape[1] / 2)
    # 首先计算抓取器两夹爪的端点
    k = math.tan(grasp_angle)
    H, W = camera_depth.shape

    grasp_width /= 2
    if k == 0:
        dx = grasp_width
        dy = 0
    else:
        dx = k / abs(k) * grasp_width / pow(k ** 2 + 1, 0.5)
        dy = k * dx
    
    pt1 = (max(min(int(grasp_row - dy), H-1), 0), max(min(int(grasp_col + dx), W-1), 0))
    pt2 = (max(min(int(grasp_row + dy), H-1), 0), max(min(int(grasp_col - dx), W-1), 0))

    # print('grasp_row, grasp_col = ', grasp_row, grasp_col)
    # print('pt1 = ', pt1[0], pt1[1])
    # print('pt2 = ', pt2[0], pt2[1])

    # cv2.circle(img_dep_rgb, (pt1[1], pt1[0]), 3, (0, 0, 0), -1) 
    # cv2.circle(img_dep_rgb, (pt2[1], pt2[0]), 3, (0, 0, 0), -1) 

    # 下面改成，从抓取线上的最高点开始向下计算抓取深度，直到碰撞或达到最大深度
    rr, cc = line(pt1[0], pt1[1], pt2[0], pt2[1])   # 获取抓取线路上的点坐标
    min_depth = np.min(camera_depth[rr, cc])
    # print('camera_depth[grasp_row, grasp_col] = ', camera_depth[grasp_row, grasp_col])
    # print('min_depth = ', min_depth)

    grasp_depth = min_depth + 0.003
    while grasp_depth < min_depth + 0.03:
        # print('--1--')
        if not collision_detection(pt1, grasp_depth, grasp_angle, camera_depth, finger_l1, finger_l2, img_dep_rgb):
            return grasp_depth - 0.003 - min_depth
        # print('--2--')
        if not collision_detection(pt2, grasp_depth, grasp_angle + math.pi, camera_depth, finger_l1, finger_l2, img_dep_rgb):
            return grasp_depth - 0.003 - min_depth
        grasp_depth += 0.003

    return grasp_depth - min_depth

def filterGrasp(grasps, img_dep, thresh=0.005):
    """
    筛选抓取配置,抓取深度超过5mm
    grasps: list([row, col, angle, width, conf])
    angle: 弧度
    width: 米
    im_dep: np.float  单位m
    """
    ret1 = []
    grasp_depths = []
    for grasp in grasps:
        row, col, angle, width, conf = grasp    # width: 单位m
        dep = img_dep[int(row), int(col)]
        # 计算 抓取深度
        finger_l1 = 0.03   # m
        finger_l2 = 0.01   # m
        grasp_depth = getGraspDepth(img_dep, row, col, angle, 
                                    length_TO_pixels(width, dep), 
                                    length_TO_pixels(finger_l1, dep), 
                                    length_TO_pixels(finger_l2, dep))
        if grasp_depth >= 0.003:
            ret1.append(grasp)
            grasp_depths.append(grasp_depth)

    if len(ret1) > 0:
        grasp_depths = np.array(grasp_depths, dtype=np.float)
        ids =  np.argsort(grasp_depths)[::-1]
        ret1 = np.array(ret1, dtype=np.float)   # (n, 5)
        ret1 = ret1[ids]
        print('return filter grasp')
        return ret1.tolist()

        # return ret1
    else:
        print('return origin grasp')
        return grasps

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

def depth2RGB(im_depth):
    """
    将深度图转至三通道8位彩色图
    先将值为0的点去除，然后转换为彩图，然后将值为0的点设为红色
    (h, w, 3)
    im_depth: 单位 mm
    """
    im_depth = depth2Gray(im_depth)
    im_color = cv2.applyColorMap(im_depth, cv2.COLORMAP_JET)
    return im_color

def length_TO_pixels(l, dep):
    """    mm mklqi
    与相机距离为dep的平面上 有条线，长l，获取这条线在图像中的像素长度
    l: m
    dep: m
    """
    f = 613.
    return l * f / dep # f为相机内参的 df/dx和df/dy的均值
