import torch
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
model = "/home/cvpr/kinova_grasp_ws/src/kinova-ros/sim_grasp/scripts/grasp_methods/ckpt/Swin2_rgb/epoch_0490_iou_0.9981.pth"

W = torch.load(model, map_location="cuda:0")
# print(W)