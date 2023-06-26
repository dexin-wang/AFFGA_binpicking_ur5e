# AFFGA_binpicking_ur5e
A ROS package for bin-picking using AFFGA-Net with ur5e.


## 1 Setup

(1) Robot: UR5e

(2) Camera: Realsense D435i

(3) Gripper: Robotiq 140-2f

(4) Notice: When the robot arm is in the watch position, the camera is ==0.6m== away from the table.


## 2 install package
(1) Download pretrained weight `affga_d.zip` from `https://drive.google.com/file/d/1wHYOYwWoLBAylW3Yf0NWJ60QbRMT_1zz/view?usp=drive_link`, unzip it and place at `scripts\grasp_methods\affga_net\pretrained_model\affga_d`.

(2) Modify camera parameters `camera_k` at `scripts\detect_grasps.py`.

(3) Modify hand_eye_calibration launch at `launch\grasp_affga.launch`, the file must publish the coordinate system of the camera `camera_color_optical_frame` with respect to the robot `base`, or you can change the coordinate system name to your own at `scripts\policy.py` (Line 73).

(4) install robotiq package: refer to https://github.com/dexin-wang/robotiq_2f_gripper_control.


## 3 run

(1) run robot, camera, robotiq, hand_eye_calibration:

> roslaunch 
