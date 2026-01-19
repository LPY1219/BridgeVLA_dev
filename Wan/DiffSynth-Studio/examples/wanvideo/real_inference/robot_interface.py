#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机器人交互接口模块
封装与真机机器人交互的所有功能，包括：
- 机械臂控制
- 相机数据获取
- 传感器数据读取
- 动作执行
"""

import time
import os
from pathlib import Path
import cv2
import numpy as np
import rospy
import sys
sys.path.append("/media/casia/data4/lpy/RoboWan")
from autolab_core import RigidTransform
from franka_interface_msgs.msg import SensorDataGroup
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto import PosePositionSensorMessage
from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional, Dict, List
from PIL import Image

# 导入相机工具
try:
    from diffusion_policy.real_world.real_camera_rs_utils import RSCapture
    from diffusion_policy.real_world.real_camera_utils import Camera
except ImportError:
    print("Warning: Could not import camera utilities. Make sure diffusion_policy is in the path.")
    RSCapture = None
    Camera = None


class RobotController:
    """机械臂控制器"""

    def __init__(self, frequency: int = 10, duration: int = 0.5, with_gripper: bool = True):
        """
        初始化机械臂控制器

        Args:
            frequency: 控制频率 (Hz)
            duration: 控制总时长
            with_gripper: 是否启用夹爪 (默认True)。Push-T任务不需要夹爪，设置为False
        """
        self.frequency = frequency
        self.duration = duration
        self.with_gripper = with_gripper

        # 初始化 ROS 和机械臂
        # init_node=False 因为ROS节点已经在外部初始化
        self.franka = FrankaArm(init_node=False, with_gripper=with_gripper)
        print("debug")
        self.rate = rospy.Rate(frequency)
        self.publisher = rospy.Publisher(
            FC.DEFAULT_SENSOR_PUBLISHER_TOPIC,
            SensorDataGroup,
            queue_size=20
        )

        # 在初始化前强制停止所有正在运行的skill
        rospy.loginfo("Stopping any running skills...")
        try:
            self.franka.stop_skill()
            rospy.sleep(0.5)  # 等待skill完全停止
            rospy.loginfo("✓ All skills stopped")
        except Exception as e:
            rospy.logwarn(f"Could not stop skills (may be none running): {e}")

        # 初始化夹爪（仅当 with_gripper=True 时）
        if self.with_gripper:
            rospy.loginfo("Initializing gripper...")
            try:
                self.franka.reset_joints()
                self.franka.open_gripper()
                rospy.sleep(1.0)
                rospy.loginfo("✓ Gripper initialized")
            except Exception as e:
                rospy.logerr(f"Gripper initialization failed: {e}")
                raise
            self.gripper_state = False  # False=开, True=闭
        else:
            rospy.loginfo("Gripper disabled (with_gripper=False)")
            # self.franka.reset_joints()
            self.gripper_state = False  # 默认状态
        self.action_id = 0

        # 获取并设置初始位姿
        curr_pose = self.franka.get_pose()
        self.franka.goto_pose(
            curr_pose,
            duration=duration,
            dynamic=True,
            buffer_time=0.1,  # 使用很短的buffer_time
            cartesian_impedances=[200.0, 200.0, 200.0, 20.0, 20.0, 20.0],  # 降低刚度
        )

    def cleanup(self):
        """
        清理机器人资源，停止所有skill
        在程序退出时调用以确保机械臂处于安全状态
        """
        rospy.loginfo("Cleaning up robot controller...")
        try:
            # 停止所有正在运行的skill
            self.franka.stop_skill()
            rospy.loginfo("✓ All skills stopped")
        except Exception as e:
            rospy.logwarn(f"Error during cleanup: {e}")

    def __del__(self):
        """析构函数，确保资源被释放"""
        try:
            self.cleanup()
        except:
            pass

    def get_pose(self) -> RigidTransform:
        """获取当前机械臂位姿"""
        return self.franka.get_pose()

    def get_gripper_width(self) -> float:
        """获取夹爪宽度"""
        return self.franka.get_gripper_width()

    def get_gripper_state(self, threshold: float = 0.01) -> bool:
        """
        获取夹爪状态

        Args:
            threshold: 判断夹爪闭合的阈值

        Returns:
            True表示闭合，False表示打开
        """
        return self.get_gripper_width() < threshold

    def open_gripper(self):
        """打开夹爪"""
        if not self.with_gripper:
            rospy.logdebug("Gripper disabled, skipping open_gripper()")
            return
        rospy.loginfo("Opening gripper...")
        self.franka.open_gripper()
        time.sleep(1.0)
        self.gripper_state = False

    def close_gripper(self):
        """闭合夹爪"""
        if not self.with_gripper:
            rospy.logdebug("Gripper disabled, skipping close_gripper()")
            return
        rospy.loginfo("Closing gripper...")
        self.franka.close_gripper()
        time.sleep(1.0)
        self.gripper_state = True

    def control_gripper(self, action: float, threshold: float = 0.5):
        """
        根据动作值控制夹爪

        Args:
            action: 夹爪动作值 (0=闭合, 1=打开)
            threshold: 判断阈值
        """
        if not self.with_gripper:
            rospy.logdebug("Gripper disabled, skipping control_gripper()")
            return
        if action < threshold and not self.gripper_state:
            rospy.loginfo(f"Closing gripper (action={action:.3f})")
            self.close_gripper()
        elif action > threshold and self.gripper_state:
            rospy.loginfo(f"Opening gripper (action={action:.3f})")
            self.open_gripper()

    def publish_pose(self, pose: RigidTransform, timestamp: float):
        """发布机械臂目标位姿"""
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=self.action_id,
            timestamp=timestamp,
            position=pose.translation,
            quaternion=pose.quaternion
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg,
                SensorDataMessageType.POSE_POSITION
            ),
        )
        self.publisher.publish(ros_msg)
        self.rate.sleep()

    def execute_pose(self, target_pose: RigidTransform, duration_s: float = 0.5):
        """
        执行目标位姿（在指定时间内持续发布）

        Args:
            target_pose: 目标位姿
            duration_s: 执行时长（秒）
        """
        t0 = time.time()
        while (time.time() - t0) < duration_s:
            self.publish_pose(target_pose, time.time())
        self.action_id += 1


# ====================== 辅助函数 ======================

def action_to_pose(action: np.ndarray,
                   current_pose: RigidTransform,
                   is_relative: bool = True) -> RigidTransform:
    """
    将动作转换为目标位姿

    Args:
        action: 动作向量 [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
                或 [x, y, z, roll, pitch, yaw, gripper]
        current_pose: 当前位姿
        is_relative: 动作是否为相对动作

    Returns:
        目标位姿
    """
    if is_relative:
        # 相对动作：增量加到当前位姿
        delta_pos = action[:3]
        delta_rot = action[3:6]  # roll, pitch, yaw

        curr_trans = current_pose.translation
        curr_rot = current_pose.rotation

        # 计算新位置
        new_trans = curr_trans + delta_pos

        # 计算新旋转
        delta_rot_mat = R.from_euler('xyz', delta_rot).as_matrix()
        new_rot = delta_rot_mat @ curr_rot

        return RigidTransform(
            rotation=new_rot,
            translation=new_trans,
            from_frame='franka_tool',
            to_frame='world'
        )
    else:
        # 绝对动作：直接使用动作值
        pos = action[:3]
        rot_euler = action[3:6]  # roll, pitch, yaw
        rot_mat = R.from_euler('xyz', rot_euler).as_matrix()

        return RigidTransform(
            rotation=rot_mat,
            translation=pos,
            from_frame='franka_tool',
            to_frame='world'
        )

