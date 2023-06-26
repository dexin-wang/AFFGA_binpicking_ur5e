import math
import cv2
import os
import zipfile
import scipy.stats as ss
import skimage.transform as skt
from random import choice
import random
import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_to_rotation_matrix(q):
    """
    四元数转旋转矩阵
    xyzw
    """
    rot_matrix = np.array(
        [[1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2])],
         [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0])],
         [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1])]],
        dtype=np.float64)
    return rot_matrix

def rotation_matrix_to_quaternion(rotMat):
    """
    旋转矩阵转四元数 xyzw
    """
    r3 = R.from_matrix(rotMat)
    qua = r3.as_quat()
    return qua


def eulerAnglesToRotationMatrix(theta):
    """
    欧拉角转旋转矩阵
    theta: [r, p, y]
    """
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                                
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R


def rotationMatrixToEulerAngles(RotM):
    """
    旋转矩阵转欧拉角
    RotateM: 旋转矩阵 ndarray shape=(3,3)
    """
    sciangle_0, sciangle_1, sciangle_2 = R.from_matrix(RotM).as_euler('xyz')

    return np.array([sciangle_0, sciangle_1, sciangle_2])

def getTransformMat(pos, rot):
    """
    将输入的pos和rot转换成 转换矩阵 格式

    input:
        pos: [ndarray, (3,), np.float]
            xyz坐标
        rot: [ndarray, (3,3), np.float]
            旋转矩阵

    return:
        T: [ndarray, (4,4), np.float]
            转换矩阵
    """
    T = np.zeros((4, 4), dtype=float)
    T[:3, :3] = rot
    T[:, 3] = np.r_[np.array(pos), np.array([1,])]
    return T

def getTransformMat1(offset, rotate):
    """
    将平移向量和旋转矩阵合并为变换矩阵
    offset: (x, y, z)
    rotate: 旋转矩阵 (9,)
    """
    mat = np.array([
        [rotate[0], rotate[1], rotate[2], offset[0]], 
        [rotate[3], rotate[4], rotate[5], offset[1]], 
        [rotate[6], rotate[7], rotate[8], offset[2]],
        [0, 0, 0, 1.] 
    ])
    return mat


def depth2Gray(im_depth):
    """
    将深度图转至8位灰度图
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    # if x_max == x_min:
    #     print('图像渲染出错 ...')
    #     raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max
    return (im_depth * k + b).astype(np.uint8)

def depth2Gray3(im_depth):
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
    ret = np.expand_dims(ret, 2).repeat(3, axis=2)
    return ret

def depth2RGB(im_depth):
    """
    将深度图转至三通道8位彩色图
    先将值为0的点去除，然后转换为彩图，然后将值为0的点设为红色
    (h, w, 3)
    im_depth: 单位 mm或m
    """
    im_depth = depth2Gray(im_depth)
    im_color = cv2.applyColorMap(im_depth, cv2.COLORMAP_JET)
    return im_color

# def resize(img):
#     returncv2.resize(img, (1000, 1000))

def distancePt(pt1, pt2):
    """
    计算两点之间的欧氏距离
    pt: [row, col] 或 [x, y]
    return: float
    """
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

def distancePt3d(pt1, pt2):
    """
    计算两点之间的欧氏距离
    pt: [x, y, z]
    return: float
    """
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2) ** 0.5


def calcAngleOfPts(pt1, pt2):
    """
    计算从pt1到pt2的逆时针夹角 [0, 2pi)
    
    pt: [x, y] 二维坐标系中的坐标，不是图像坐标系的坐标
    
    return: 弧度
    """
    dy = pt2[1] - pt1[1]
    dx = pt2[0] - pt1[0]
    return (math.atan2(dy, dx) + 2*math.pi) % (2*math.pi)
    

def radians_TO_angle(radians):
    """
    弧度转角度
    """
    return 180 * radians / math.pi

def angle_TO_radians(angle):
    """
    角度转弧度
    """
    return math.pi * angle / 180

def depth3C(depth):
    """
    将深度图转化为3通道 np.uint8类型
    """
    depth_3c = depth[..., np.newaxis]
    depth_3c = np.concatenate((depth_3c, depth_3c, depth_3c), axis=2)
    return depth_3c.astype(np.uint8)

def zip_file(filedir):
    """
    压缩文件夹至同名zip文件
    """
    file_news = filedir + '.zip'
    # 如果已经有压缩的文件，删除
    if os.path.exists(file_news):
        os.remove(file_news)

    z = zipfile.ZipFile(file_news,'w',zipfile.ZIP_DEFLATED) #参数一：文件夹名
    for dirpath, dirnames, filenames in os.walk(filedir):
        fpath = dirpath.replace(filedir,'') #这一句很重要，不replace的话，就从根目录开始复制
        fpath = fpath and fpath + os.sep or ''#这句话理解我也点郁闷，实现当前文件夹以及包含的所有文件的压缩
        for filename in filenames:
            z.write(os.path.join(dirpath, filename),fpath+filename)
    z.close()


def unzip(file_name):
    """
    解压缩zip文件至同名文件夹
    """
    zip_ref = zipfile.ZipFile(file_name) # 创建zip 对象
    os.mkdir(file_name.replace(".zip","")) # 创建同名子文件夹
    zip_ref.extractall(file_name.replace(".zip","")) # 解压zip文件内容到子文件夹
    zip_ref.close() # 关闭zip文件

def imresize(image, size, interp="nearest"):
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic'"
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation '{}' not"
                                      " supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size '{}'.".format(type(size)))

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")

def gaussian_noise(im_depth):
    """
    在image上添加高斯噪声，参考dex-net代码

    im_depth: 浮点型深度图，单位为米
    """
    gamma_shape = 1000.00
    gamma_scale = 1 / gamma_shape
    # gaussian_process_sigma = choice([0.002, 0.003, 0.004])   # 0.004
    gaussian_process_sigma = 0.004   # 默认 0.004
    gaussian_process_scaling_factor = 8.0

    im_height, im_width = im_depth.shape
    
    # 1
    mult_samples = ss.gamma.rvs(gamma_shape, scale=gamma_scale, size=1) # 生成一个接近1的随机数，shape=(1,)
    mult_samples = mult_samples[:, np.newaxis]
    im_depth = im_depth * np.tile(mult_samples, [im_height, im_width])  # 把mult_samples复制扩展为和camera_depth同尺寸，然后相乘
    
    # 2
    gp_rescale_factor = gaussian_process_scaling_factor     # 4.0
    gp_sample_height = int(im_height / gp_rescale_factor)   # im_height / 4.0
    gp_sample_width = int(im_width / gp_rescale_factor)     # im_width / 4.0
    gp_num_pix = gp_sample_height * gp_sample_width     # im_height * im_width / 16.0
    gp_sigma = gaussian_process_sigma

    gp_noise = ss.norm.rvs(scale=gp_sigma, size=gp_num_pix).reshape(gp_sample_height, gp_sample_width)  # 生成(均值为0，方差为scale)的gp_num_pix个数，并reshape
    # print('高斯噪声最大误差:', gp_noise.max())
    gp_noise = imresize(gp_noise, gp_rescale_factor, interp="bicubic")  # resize成图像尺寸，bicubic为双三次插值算法
    # gp_noise[gp_noise < 0] = 0
    # camera_depth[camera_depth > 0] += gp_noise[camera_depth > 0]
    im_depth += gp_noise

    # print(gp_noise.min(), gp_noise.max())

    return im_depth

def add_missing_val(im_depth):
    """
    添加缺失值
    梯度大的位置，概率大

    im_depth: 浮点型深度图，单位为米
    """
    global grad
    # 获取梯度图
    ksize = 11   # 调整ksize 11
    grad_X = np.abs(cv2.Sobel(im_depth, -1, 1, 0, ksize=ksize))    
    grad_Y = np.abs(cv2.Sobel(im_depth, -1, 0, 1, ksize=ksize))
    grad = cv2.addWeighted(grad_X, 0.5, grad_Y, 0.5, 0)
    
    # cv2.imshow('gradient', tool.depth2Gray(grad))
    # cv2.imwrite('D:/research/grasp_detection/sim_grasp/realsense/grad11.png', tool.depth2Gray(grad))

    # 产生缺失值的概率与梯度值呈正比
    im_height, im_width = im_depth.shape
    gp_rescale_factor = 8.0     # 缩放尺度
    gp_sample_height = int(im_height / gp_rescale_factor)
    gp_sample_width = int(im_width / gp_rescale_factor)

    # 计算梯度对应的概率，线性方程
    """
    Sobel ksize 与 g1 g2 的对应关系
    ksize = 11  --> 300.0  3000.0
    ksize = 7  --> 1  30
    """
    min_p = 0.001
    max_p = 1.0
    g1, p1 = 500.0, min_p
    g2, p2 = 1500.0, max_p
    prob_k = (p2 - p1) / (g2 - g1)
    prob_b = p2 - prob_k * g2

    prob = grad * prob_k + prob_b
    prob = np.clip(prob, 0, max_p)

    # 生成0-1随机数矩阵
    random_mat = np.random.rand(gp_sample_height, gp_sample_width)
    random_mat = imresize(random_mat, gp_rescale_factor, interp="bilinear")  # 放大
    # cv2.imshow('random_mat', tool.depth2Gray(random_mat))

    # random_mat小于prob的位置缺失
    im_depth[ random_mat < prob ] = 0.0

    return im_depth

def slope(img):
    """
    给输入图像加入斜坡函数

    img: np.array shape=(H, W)
    
    即生成一个shape与img相同，但值为倾斜桌面深度值的二维数组
    二维数组值的方程:z = a1 * (x - W/2) + a2 * (y - H/2) + c
    """
    H, W = img.shape
    X = np.arange(0, W)
    Y = np.arange(0, H)
    X, Y = np.meshgrid(X, Y)

    a1 = random.random() * 0.0002   # 默认0.0001
    a2 = random.random() * 0.0002

    Z = a1 * (X - W/2 + 0.5) + a2 * (Y - H/2 + 0.5)
    # print(Z.shape)
    img = img + Z
    return img

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

# def add_noise_depth_experiment(img):
#     """
#     为深度图像添加噪声(在线验证系统用)
#     """
#     img = add_missing_val(img)  # 缺失值
#     img = inpaint(img, missing_value=0) # 补全
#     img = gaussian_noise(img)   # 高斯噪声
#     # 模糊
#     for i in range(20):
#         img = imresize(img, 0.5, interp='bilinear')
#         img = imresize(img, 2.0, interp='bilinear')
#     # 倾斜
#     # img = slope(img)
#     return img

def add_noise_depth_experiment(img):
    """
    为深度图像添加噪声(在线验证系统用)
    """
    # 倾斜
    img = slope(img)
    img = gaussian_noise(img)   # 高斯噪声
    # 模糊
    # for i in range(5):
    #     img = imresize(img, 0.25, interp='bilinear')
    #     img = imresize(img, 4.0, interp='bilinear')
    for i in range(10):
        img = imresize(img, 0.5, interp='bilinear')
        img = imresize(img, 2.0, interp='bilinear')
    img = add_missing_val(img)  # 缺失值
    img = inpaint(img, missing_value=0) # 补全
    
    return img

def add_noise_depth(img, n=None):
    """
    为深度图像添加噪声
    """
    # 模糊
    # num = choice([10, 20]) if n is None else n
    num = 20
    for i in range(num):
        img = imresize(img, 0.5, interp='bilinear')
        img = imresize(img, 2.0, interp='bilinear')

    img = gaussian_noise(img)   # 高斯噪声

    img = add_missing_val(img)  # 缺失值
    img = inpaint(img, missing_value=0) # 补全
    
    # 倾斜
    # img = slope(img)
    return img

def _Hue(img, bHue, gHue, rHue):
    # 1.计算三通道灰度平均值
    imgB = img[:, :, 0]
    imgG = img[:, :, 1]
    imgR = img[:, :, 2]

    # 下述3行代码控制白平衡或者冷暖色调，下例中增加了b的分量，会生成冷色调的图像，
    # 如要实现白平衡，则把两个+10都去掉；如要生成暖色调，则增加r的分量即可。
    bAve = cv2.mean(imgB)[0] + bHue
    gAve = cv2.mean(imgG)[0] + gHue
    rAve = cv2.mean(imgR)[0] + rHue
    aveGray = (int)(bAve + gAve + rAve) / 3

    # 2计算每个通道的增益系数
    bCoef = aveGray / bAve
    gCoef = aveGray / gAve
    rCoef = aveGray / rAve

    # 3使用增益系数
    imgB = np.expand_dims(np.floor((imgB * bCoef)), axis=2)
    imgG = np.expand_dims(np.floor((imgG * gCoef)), axis=2)
    imgR = np.expand_dims(np.floor((imgR * rCoef)), axis=2)

    dst = np.concatenate((imgB, imgG, imgR), axis=2)
    dst = np.clip(dst, 0, 255).astype(np.uint8)
    return dst

def add_noise_rgb(img, hue=10):
    """
    为彩色图像添加噪声
    """
    # 调节色调
    hue = np.random.uniform(-1 * hue, hue)

    if hue == 0:
        # 一般的概率保持原样 / 白平衡
        if np.random.rand() < 0.5:
            # 白平衡
            img = _Hue(img, hue, hue, hue)
    else:
        # 冷暖色调
        bHue = hue if hue > 0 else 0
        gHue = abs(hue)
        rHue = -1 * hue if hue < 0 else 0
        img = _Hue(img, bHue, gHue, rHue)

    # 调节亮度
    bright = np.random.uniform(-40, 10)
    imgZero = np.zeros(img.shape, img.dtype)
    img = cv2.addWeighted(img, 1, imgZero, 2, bright)
    return img


if __name__ == "__main__":
    qua = np.array([0, 0, 0, 1])
    r = quaternion_to_rotation_matrix(qua)
    q = rotation_matrix_to_quaternion(r)
    print(q)
