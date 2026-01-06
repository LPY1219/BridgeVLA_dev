#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import io
import termios
# 强制设置标准输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def flush_input_buffer():
    """清空stdin输入缓冲区，防止之前的按键影响后续input()"""
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except:
        pass

"""
用于对机械臂遥操作并保存数据的代码进行Debug
"""
import sys
import os
import time
import pickle as pkl
import numpy as np
from typing import Dict, List, Optional, Any

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'input_device'))
sys.path.insert(0, os.path.join(current_dir, 'shared_memory'))


# 导入依赖
import rospy
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.proto import PosePositionSensorMessage
from scipy.spatial.transform import Rotation as R
from multiprocessing.managers import SharedMemoryManager
import cv2
from real_camera_utils_lpy import get_cam_extrinsic


# 导入自定义模块
try:
   from pynput import keyboard as pynput_keyboard
   pynput_available = True
except ImportError:
   print("Warning: pynput not installed. Please install: pip install pynput")
   pynput_keyboard = None
   pynput_available = False


from input_device.spacemouse import FrankaSpacemouse
from shared_memory.shared_memory_queue import SharedMemoryQueue
from shared_memory.shared_ndarray import SharedNDArray
from real_camera_utils_lpy import Camera
from data_collection_utils import pose2array, motion2array, publish_pose


class CollectDataWithTeleop2:
   """遥操作控制器 + img_rgb + depth + 机械臂目标状态采集"""
  
   def __init__(self, frequency: float = 30.0, duration:float = 60.0, task_name:str = 'debug1', trail:int = 0, gripper_thres:float = 0.05, instruction:str = "place the block on the plate", save_interval: int = 1, resolution: str = "HD1080", camera=None, fa=None):
       """
           Args:
               frequency: 目标的采集频率
               duration: 采集动作时长
               task_name: 任务名称
               gripper_thres: 夹爪阈值
               instruction: 任务描述
               save_interval: 保存间隔，每N步保存一次数据（默认每步都保存）
               resolution: 图像分辨率类型，可选 "HD1080" 或 "VGA"
               camera: 可选的相机对象（用于复用已初始化的相机）
               fa: 可选的机械臂对象（用于复用已初始化的机械臂）
       """
       self.task_name = task_name
       self.trail = trail
       self.frequency = frequency
       self.dt = 1.0 / frequency
       self.duration = duration
       self.total_steps = int(duration * frequency)
       self.gripper_thres = gripper_thres
       self.instruction = instruction
       self.save_interval = save_interval  # 保存间隔

       # 设置图像分辨率
       self.resolution = resolution
       if resolution == "HD1080":
           self.img_height = 1080
           self.img_width = 1920
       elif resolution == "VGA":
           self.img_height = 376
           self.img_width = 672
       else:
           raise ValueError(f"不支持的分辨率类型: {resolution}，请使用 'HD1080' 或 'VGA'")

       # 初始化或复用机械臂
       if fa is None:
           rospy.loginfo("[Robot] 初始化机械臂...")
           self.fa = FrankaArm()
           self.fa.reset_joints()
           self.fa_is_owned = True  # 标记机械臂是否由此对象创建
       else:
           rospy.loginfo("[Robot] 复用已初始化的机械臂...")
           self.fa = fa
           self.fa_is_owned = False

       self.current_pose = self.fa.get_pose()
       # 暂存的目标动作，用于后续通过ros向机械臂传输动作
       self.target_pose = self.current_pose.copy()

       # 初始化或复用相机
       if camera is None:
           rospy.loginfo(f"[Camera] 初始化3个第三视角Zed相机 ({self.resolution}: {self.img_height}x{self.img_width})...")
           self.camera = Camera(camera_type="all", zed_resolution=self.resolution)
           self.camera_is_owned = True  # 标记相机是否由此对象创建
       else:
           rospy.loginfo(f"[Camera] 复用已初始化的相机 ({self.resolution}: {self.img_height}x{self.img_width})...")
           self.camera = camera
           self.camera_is_owned = False
      
       # ROS相关内容
       self.rate = rospy.Rate(frequency)
       self.publisher = rospy.Publisher(
           FC.DEFAULT_SENSOR_PUBLISHER_TOPIC,
           SensorDataGroup,
           queue_size = 20
       )
      
       #* 记录状态变量
       self.init_time = None
       self.step_counter = 0
       self.control_step_counter = 0  # 控制步数计数器（用于计算保存间隔）
       self.recording = False
       self.gripper_control_in_progress = False  # 夹爪控制进行中标志
       self.should_exit = False  # 退出标志
      
       #* 键盘状态
       self.keys_pressed = set()
       self.gripper_state = False # False夹爪打开，True夹爪闭合
       self.last_g_state = False
       self.last_r_state = False
      
       # 初始化夹爪状态（只在第一次初始化机械臂时执行）
       if self.fa_is_owned:
           try:
               rospy.loginfo("[Setup] 初始化夹爪状态...")
               self.fa.open_gripper()  # 确保夹爪处于打开状态
               rospy.sleep(1.0)  # 等待夹爪动作完成
               rospy.loginfo("[OK] 夹爪初始化完成（打开状态）")
           except Exception as e:
               rospy.logerr(f"夹爪初始化失败: {e}")
       else:
           rospy.loginfo("[Reuse] 复用机械臂，跳过夹爪初始化")
      
       #* 数据存储功能
       self.data_arrays: Dict[str, SharedNDArray] = {} # 存储机械臂状态相关内容
      
   def setup_shared_arrays(self, shm_manager: SharedMemoryManager):
       """设置共享内存数组"""
       rospy.loginfo(f"{'=' * 20} 正在设置共享内存数组 {'=' * 20}")
       rospy.loginfo(f"[Resolution] 使用分辨率: {self.resolution} ({self.img_height}x{self.img_width})")

       # 图像大小 - 动态设置根据分辨率配置
       bgr_shape = (self.total_steps, self.img_height, self.img_width, 3)
       depth_shape = (self.total_steps, self.img_height, self.img_width)
       pcd_shape = (self.total_steps, self.img_height, self.img_width, 3)

       # 保存三个第三视角相机的图像BGR
       self.data_arrays['3rd_1_bgr_images'] = SharedNDArray.create_from_shape(
           shm_manager, bgr_shape, np.uint8
       ) # 0-255
       self.data_arrays['3rd_2_bgr_images'] = SharedNDArray.create_from_shape(
           shm_manager, bgr_shape, np.uint8
       ) # 0-255
       self.data_arrays['3rd_3_bgr_images'] = SharedNDArray.create_from_shape(
           shm_manager, bgr_shape, np.uint8
       ) # 0-255

       # 记录机械臂位姿 xyz + quat
       self.data_arrays['poses'] = SharedNDArray.create_from_shape(
           shm_manager, (self.total_steps, 7), np.float32
       ) # 机械臂位姿 [x,y,z,qw,qx,qy,qz] 0-1

       self.data_arrays['gripper_states'] = SharedNDArray.create_from_shape(
           shm_manager, (self.total_steps,), np.bool_
       ) # 夹爪状态

       # 三个相机的深度图
       self.data_arrays['3rd_1_depth'] = SharedNDArray.create_from_shape(
           shm_manager, depth_shape, np.float32
       )
       self.data_arrays['3rd_2_depth'] = SharedNDArray.create_from_shape(
           shm_manager, depth_shape, np.float32
       )
       self.data_arrays['3rd_3_depth'] = SharedNDArray.create_from_shape(
           shm_manager, depth_shape, np.float32
       )

       # 三个相机的点云数据
       self.data_arrays['3rd_1_pcd'] = SharedNDArray.create_from_shape(
           shm_manager, pcd_shape, np.float32
       )
       self.data_arrays['3rd_2_pcd'] = SharedNDArray.create_from_shape(
           shm_manager, pcd_shape, np.float32
       )
       self.data_arrays['3rd_3_pcd'] = SharedNDArray.create_from_shape(
           shm_manager, pcd_shape, np.float32
       )

       self.data_arrays['joints'] = SharedNDArray.create_from_shape(
              shm_manager, (self.total_steps, 7), np.float32
         )

       rospy.loginfo(f"[OK] 共享数组创建完成，预分配 {self.total_steps} 个数据点")
      
  
   def update_keyboard_state(self) -> Dict[str, Any]:
       """更新键盘状态"""
       if not pynput_available:
           return {'gripper_state': self.gripper_state, 'recording': self.recording}
          
       current_g = 'g' in self.keys_pressed
       current_r = 'r' in self.keys_pressed
      
       # 添加调试信息
       if len(self.keys_pressed) > 0 and (current_g or current_r):
           rospy.loginfo(f"当前按下的键: {self.keys_pressed}")
      
       # Toggle模式
       if current_g and not self.last_g_state and not self.gripper_control_in_progress:
           self.gripper_control_in_progress = True  # 设置夹爪控制进行中标志
           self.gripper_state = not self.gripper_state

           # 执行夹爪动作（阻塞控制）
           try:
               if self.gripper_state:  # True = 夹爪关闭
                   print("🤏 夹爪关闭中...")
                   self.fa.close_gripper()
               else:  # False = 夹爪打开
                   print("✋ 夹爪打开中...")
                   self.fa.open_gripper()

               # 等待夹爪动作完成（阻塞控制）
               rospy.sleep(1.0)  # 给夹爪足够时间完成动作
               print("✅ 夹爪动作完成")
              
           except Exception as e:
               rospy.logerr(f"夹爪控制出错: {e}")
               # 如果夹爪控制失败，恢复之前的状态
               self.gripper_state = not self.gripper_state
               rospy.logwarn(f"夹爪状态已恢复为: {'闭合' if self.gripper_state else '打开'}")
           finally:
               self.gripper_control_in_progress = False  # 清除夹爪控制进行中标志
          
       if current_r and not self.last_r_state:
           if self.recording:
               # 停止录制并设置退出标志
               self.recording = False
               self.should_exit = True
               print("\n" + "🛑"*20, flush=True)
               print("⏹️  停止录制，正在保存数据...", flush=True)
               print("🛑"*20 + "\n", flush=True)
           else:
               # 开始录制
               self.recording = True
               print("\n" + "🔴"*20, flush=True)
               print("🎬 检测到 [R] 键！开始录制！", flush=True)
               print("▶️  移动机械臂进行操作...", flush=True)
               print("🔴"*20 + "\n", flush=True)
      
       self.last_g_state = current_g
       self.last_r_state = current_r
      
       return {'gripper_state': self.gripper_state, 'recording': self.recording}
  
   def on_key_press(self, key):
       """按键按下回调"""
       try:
           if hasattr(key, 'char') and key.char is not None:
               self.keys_pressed.add(key.char.lower())
           elif hasattr(key, 'name'):
               self.keys_pressed.add(key.name.lower())
       except Exception as e:
           rospy.logwarn(f"键盘按键处理错误: {e}")
      
   def on_key_release(self, key):
       """按键释放回调"""
       try:
           if hasattr(key, 'char') and key.char is not None:
               self.keys_pressed.discard(key.char.lower())
           elif hasattr(key, 'name'):
               self.keys_pressed.discard(key.name.lower())
       except Exception as e:
           rospy.logwarn(f"键盘按键处理错误: {e}")
      
   def control_step(self) -> bool:
    """控制步骤：采集相机并写入共享数组 -> 写入机械臂状态 -> 发布控制由主循环完成"""
    # 注意：键盘状态已在主循环中更新，这里不再重复调用

    # 未开启录制：直接返回
    if not self.recording:
        return True

    # ---- 2) 相机采集 ----
    step_start = time.time()

    # 三个第三视角相机（ZED）通过已有的封装
    # result_dict['3rd_1'], result_dict['3rd_2'], result_dict['3rd_3']
    # 每个包含: {'rgb': (H,W,3) BGR, 'depth': (H,W) float32, 'pcd': (H,W,3) float32}
    result_dict = self.camera.capture()

    capture_time = time.time() - step_start

    # ---- 3) 初始化起始时间 ----
    if self.init_time is None:
        self.init_time = time.time()

    # ---- 4) 按保存间隔写入共享数组 ----
    time_before_save = time.time()
    do_save = (self.step_counter < self.total_steps) and (self.control_step_counter % self.save_interval == 0)

    if do_save:
        # 优化：一次性获取所有数组引用，避免重复调用.get()
        idx = self.step_counter

        # 4.1 三个第三视角相机 - BGR 图像 (优化版)
        time_bgr_start = time.time()
        for cam_type in ['3rd_1', '3rd_2', '3rd_3']:
            key = f'{cam_type}_bgr_images'
            if key in self.data_arrays:
                self.data_arrays[key].get()[idx] = result_dict[cam_type]['rgb']
        time_bgr = time.time() - time_bgr_start

        # 4.2 三个第三视角相机 - 深度与点云 (优化版)
        time_depth_start = time.time()
        for cam_type in ['3rd_1', '3rd_2', '3rd_3']:
            depth_key = f'{cam_type}_depth'
            pcd_key = f'{cam_type}_pcd'
            if depth_key in self.data_arrays:
                self.data_arrays[depth_key].get()[idx] = result_dict[cam_type]['depth']
            if pcd_key in self.data_arrays:
                self.data_arrays[pcd_key].get()[idx] = result_dict[cam_type]['pcd']
        time_depth = time.time() - time_depth_start

        # 4.3 机械臂实际状态（与图像同步写）
        time_robot_start = time.time()
        actual_pose = self.fa.get_pose()
        actual_joints = self.fa.get_joints()
        if 'poses' in self.data_arrays:
            self.data_arrays['poses'].get()[idx] = pose2array(actual_pose)
        if 'joints' in self.data_arrays:
            self.data_arrays['joints'].get()[idx] = np.array(actual_joints, dtype=np.float32)
        if 'gripper_states' in self.data_arrays:
            self.data_arrays['gripper_states'].get()[idx] = self.gripper_state
        time_robot = time.time() - time_robot_start

        time_after_save = time.time()
        total_save_time = time_after_save - time_before_save

        # 4.4 详细性能日志（每30步）
        if self.step_counter % 30 == 0:
            rospy.loginfo(f"[性能] Step {self.step_counter}: "
                          f"Capture={capture_time*1000:.1f}ms, "
                          f"Save Total={total_save_time*1000:.1f}ms "
                          f"(BGR={time_bgr*1000:.1f}ms, Depth+PCD={time_depth*1000:.1f}ms, Robot={time_robot*1000:.1f}ms)")
    else:
        # 不保存时的性能日志（每 30 个控制步打印一次）
        if self.control_step_counter % 30 == 0:
            rospy.loginfo(f"Control Step {self.control_step_counter}, Capture: {capture_time*1000:.1f}ms, Skip Save")

    # ---- 5) 一切正常，返回 True 让主循环继续 ----
    return True
  
   def run_data_collection(self, save_dir: str = "./teleop_data", trail:int = 0):
       """运行数据采集"""
       rospy.loginfo(f"[Start] 开始高频数据采集 - {self.frequency}Hz, {self.duration}s")


       with SharedMemoryManager() as shm_manager:
           # 设置共享内存
           self.setup_shared_arrays(shm_manager)
          
           # 创建Spacemouse控制器
           spacemouse = FrankaSpacemouse(
               shm_manager,
               frequency=self.frequency,
               deadzone=0.05,
               position_sensitivity=0.5, #TODO 在这里改变灵敏度
               rotation_sensitivity=0.8,
               debug=False
           )
          
           # 启动键盘监听
           keyboard_listener = None
           if pynput_available:
               keyboard_listener = pynput_keyboard.Listener(
                   on_press=self.on_key_press,
                   on_release=self.on_key_release
               )
               keyboard_listener.start()
               rospy.loginfo("[OK] 键盘监听已启动")

           rospy.loginfo("[Control] 控制说明:")
           rospy.loginfo("  - SpaceMouse: 控制机械臂移动")
           rospy.loginfo("  - 'R' 键: 开始/停止录制")
           rospy.loginfo("  - 'G' 键: 切换夹爪")
           rospy.loginfo("  - Ctrl+C: 停止采集")
           rospy.loginfo("="*50)
          
           with spacemouse:
               try:
                   # 启动机械臂动态控制
                   self.fa.goto_pose(
                       self.target_pose,
                       duration=self.duration,
                       dynamic=True,
                       buffer_time=10,
                       cartesian_impedances=[600.0, 600.0, 600.0, 50.0, 50.0, 50.0] # TODO 可以在这里降低阻抗以提高响应速度
                   )
                  
                   start_time = time.time()
                   rospy.loginfo("🔄 开始控制循环，正确观察-动作同步")
                   for i in range(self.total_steps):
                       loop_start = time.time()
                      
                       #* === 步骤1: 读取输入 (~1ms) ===
                       # 更新键盘状态（无论是否录制都要检查）
                       self.update_keyboard_state()
                      
                       motion = spacemouse.get_motion_state()
                      
                       # 调整运动方向
                       motion[0] = -motion[0]
                       motion[4] = -motion[4]
                       motion[3], motion[4] = motion[4], motion[3]
                      
                       #* === 步骤2: 计算机械臂增量 ===
                       translation_delta = motion[:3] * self.dt
                       rotation_angles = motion[3:] * self.dt
                      
                       #* === 步骤3: 将位姿增量添加到目标位姿上，这时候机械臂还没有进行移动 ===
                       self.target_pose.translation += translation_delta
                      
                       if np.linalg.norm(rotation_angles) > 1e-6:
                           rotation_scipy = R.from_euler('xyz', rotation_angles)
                           rotation_matrix_delta = rotation_scipy.as_matrix()
                           self.target_pose.rotation = self.target_pose.rotation @ rotation_matrix_delta
                      
                       #* === 步骤4: 记录相机照片以及基于目标位姿的机械臂状态 ===
                       # 只有在录制状态下且夹爪控制不在进行中时才进行数据采集
                       if self.recording and not self.gripper_control_in_progress:
                           self.control_step()
                      
                       #* === 步骤5: 发布控制指令 ===
                       # 只有在录制状态下且夹爪控制不在进行中时才发布控制指令
                       if self.recording and not self.gripper_control_in_progress and i > 0:
                           timestamp = time.time() - start_time
                           publish_pose(
                               self.target_pose,
                               i,
                               timestamp,
                               pub=self.publisher,
                               rate=self.rate
                           )
                      
                       # 增加控制步数计数器（无论是否保存数据）
                       if self.recording and not self.gripper_control_in_progress:
                           self.control_step_counter += 1
                          
                           # 只有在保存间隔时才增加数据步数计数器
                           if self.control_step_counter % self.save_interval == 0:
                               self.step_counter += 1
                      
                       #* === 步骤6: 检查退出条件 ===
                       if self.should_exit:
                           rospy.loginfo(f"[STOP] 录制已停止，当前步数: {self.step_counter}")
                           rospy.loginfo("[Terminate] 发送终止信号，停止机械臂控制...")
                           # 停止机械臂控制
                           try:
                               self.fa.stop_skill()
                               rospy.loginfo("[OK] 机械臂控制已停止")
                           except Exception as e:
                               rospy.logerr(f"停止机械臂控制时出错: {e}")
                           break
                      
                       #* === 步骤7: 频率控制 ===
                       elapsed = time.time() - loop_start
                       sleep_time = max(0, self.dt - elapsed)
                      
                       #* 性能监控
                       if i % 60 == 0:
                           if self.should_exit:
                               status = "[STOP] 准备退出"
                           elif self.gripper_control_in_progress:
                               status = "[GRIPPER] 夹爪控制中"
                           elif self.recording:
                               status = "[REC] 录制中"
                           else:
                               status = "[PAUSE] 暂停中"
                           rospy.loginfo(f"{status} - 第 {i} 步: {elapsed*1000:.1f}ms (target: {self.dt*1000:.1f}ms), 控制步: {self.control_step_counter}, 已记录: {self.step_counter} 步")
                      
                       if sleep_time > 0:
                           time.sleep(sleep_time)
                       elif elapsed > self.dt * 1.2:
                        #    rospy.logwarn(f"拍照 + 控制循环 超时: 第 {i} 步: {elapsed*1000:.1f}ms")
                        pass
                      
                       if time.time() - start_time > self.duration:
                           rospy.loginfo("[Done] 采集完成，保存数据...")
                           break

               except KeyboardInterrupt:
                   rospy.loginfo("[Interrupt] 用户中断，停止采集")
               except Exception as e:
                   rospy.logerr(f"采集错误: {e}")
                   import traceback
                   traceback.print_exc()
               finally:
                   # 停止机械臂
                   try:
                       self.fa.stop_skill()
                   except:
                       pass
                  
                   # 停止键盘监听
                   if keyboard_listener:
                       keyboard_listener.stop()

                   rospy.loginfo("Data collection ended")
                  
                   # 保存数据
                   if self.step_counter > 0:
                       rospy.loginfo(f"[Save] 保存 {self.step_counter} 步数据...")
                       self.save_collected_data(save_dir)
                   else:
                       rospy.logwarn("没有数据需要保存")

   def save_collected_data(self, save_dir: str):
    rospy.loginfo("[Save] 保存数据中...")

    trail_dir = os.path.join(save_dir, self.task_name, f"trail_{self.trail}")

    # 检查目录是否已存在
    if os.path.exists(trail_dir):
        print(f"\n⚠️  警告: 目录 {trail_dir} 已存在!", flush=True)
        while True:
            flush_input_buffer()  # 清空输入缓冲区
            overwrite = input("是否覆盖? (Y覆盖/N取消): ").strip().lower()
            if overwrite == 'y' or overwrite == 'yes':
                # 删除旧目录
                import shutil
                shutil.rmtree(trail_dir)
                print("🗑️  已删除旧数据", flush=True)
                break
            elif overwrite == 'n' or overwrite == 'no':
                print("❌ 取消保存，数据未保存", flush=True)
                return
            else:
                print(f"❓ 无效输入 '{overwrite}'，请输入 Y 或 N", flush=True)

    os.makedirs(trail_dir, exist_ok=True)
    actual_length = self.step_counter
    if actual_length == 0:
        rospy.logwarn("未采集到数据")
        return

    # --- 取出三个第三视角相机的数据 ---
    data_dict = {
        '3rd_1_bgr_images': self.data_arrays['3rd_1_bgr_images'].get()[:actual_length].copy(),
        '3rd_2_bgr_images': self.data_arrays['3rd_2_bgr_images'].get()[:actual_length].copy(),
        '3rd_3_bgr_images': self.data_arrays['3rd_3_bgr_images'].get()[:actual_length].copy(),
        '3rd_1_depth': self.data_arrays['3rd_1_depth'].get()[:actual_length].copy(),
        '3rd_2_depth': self.data_arrays['3rd_2_depth'].get()[:actual_length].copy(),
        '3rd_3_depth': self.data_arrays['3rd_3_depth'].get()[:actual_length].copy(),
        '3rd_1_pcd': self.data_arrays['3rd_1_pcd'].get()[:actual_length].copy(),
        '3rd_2_pcd': self.data_arrays['3rd_2_pcd'].get()[:actual_length].copy(),
        '3rd_3_pcd': self.data_arrays['3rd_3_pcd'].get()[:actual_length].copy(),
        'poses': self.data_arrays['poses'].get()[:actual_length].copy(),
        'gripper_states': self.data_arrays['gripper_states'].get()[:actual_length].copy(),
        'joints': self.data_arrays['joints'].get()[:actual_length].copy()
    }

    dir_names = ['3rd_1_bgr_images', '3rd_1_bgr', '3rd_2_bgr_images', '3rd_2_bgr',
                 '3rd_3_bgr_images', '3rd_3_bgr',
                 '3rd_1_depth', '3rd_2_depth', '3rd_3_depth',
                 '3rd_1_pcd', '3rd_2_pcd', '3rd_3_pcd',
                 'poses', 'gripper_states', 'joints']

    # 保存指令
    with open(os.path.join(save_dir, self.task_name, f"trail_{self.trail}", "instruction.txt"), 'w') as f:
        f.write(self.instruction)

    # 保存三个相机的外参矩阵
    extrinsics = {
        '3rd_1': get_cam_extrinsic("3rd_1"),
        '3rd_2': get_cam_extrinsic("3rd_2"),
        '3rd_3': get_cam_extrinsic("3rd_3")
    }
    with open(os.path.join(save_dir, self.task_name, f"trail_{self.trail}", "extrinsics.pkl"), 'wb') as f:
        pkl.dump(extrinsics, f, protocol=pkl.HIGHEST_PROTOCOL)

    for dir_name in dir_names:
        dir_path = os.path.join(save_dir, self.task_name, f"trail_{self.trail}", dir_name)
        os.makedirs(dir_path, exist_ok=True)

        # 保存图像文件（PNG格式）
        if dir_name in ['3rd_1_bgr_images', '3rd_2_bgr_images', '3rd_3_bgr_images']:
            print(f"正在保存 {dir_name} 图像文件")
            for i in range(actual_length):
                img_path = os.path.join(dir_path, f"{i:06d}.png")
                cv2.imwrite(img_path, data_dict[dir_name][i])

        # 保存BGR数组（PKL格式）
        elif dir_name in ['3rd_1_bgr', '3rd_2_bgr', '3rd_3_bgr']:
            print(f"正在保存 {dir_name}（BGR数组pkl）")
            # 从对应的images数据中取
            source_key = dir_name.replace('_bgr', '_bgr_images')
            for i in range(actual_length):
                with open(os.path.join(dir_path, f"{i:06d}.pkl"), 'wb') as f:
                    pkl.dump(data_dict[source_key][i], f, protocol=pkl.HIGHEST_PROTOCOL)

        # 保存其他数组数据
        else:
            print(f"正在保存 {dir_name} 数组")
            for i in range(actual_length):
                with open(os.path.join(dir_path, f"{i:06d}.pkl"), 'wb') as f:
                    pkl.dump(data_dict[dir_name][i], f, protocol=pkl.HIGHEST_PROTOCOL)

    rospy.loginfo(f"数据已保存到: {save_dir}")

      
def main():
    """主函数 - 支持连续采集多条轨迹"""
    frequency = 80.0  # 控制频率：60Hz（从80Hz降低以适应三相机采集）
    duration=600
    # task_name = 'put_lion_on_top_shelf'
    task_name = 'put_the_lion_on_the_top_shelf_20260105'
    gripper_thres = 0.05
    # instruction = "put the lion on the top shelf"
    instruction = "put the lion on the top shelf"
    task_idx =10  # 起始轨迹序号 # TODO

    data_result_dir = "/media/casia/data4/lpy/3zed_data/raw_data_5" # TODO
    save_interval = 3  # 每1步保存一次数据（即60/3=20Hz保存频率）
    resolution = "VGA"  # 图像分辨率：可选 "HD1080" (1080x1920) 或 "VGA" (376x672)

    # 在循环外初始化相机和机械臂（只初始化一次）
    # 注意：必须先初始化 FrankaArm，因为它会初始化 ROS 节点
    print("[Setup] 初始化共享资源（相机和机械臂）...")
    print("[Robot] 初始化机械臂...")
    shared_fa = FrankaArm()
    shared_fa.reset_joints()

    # 现在 ROS 节点已经由 FrankaArm 初始化，可以使用 rospy.loginfo 了
    rospy.loginfo("[Start] 启动连续采集模式 - 高频遥操作数据采集系统")
    rospy.loginfo(f"配置: {frequency}Hz控制, {frequency/save_interval}Hz保存, {duration}s, 分辨率: {resolution}")
    rospy.loginfo(f"起始轨迹序号: trail_{task_idx}")
    rospy.loginfo("="*60)

    rospy.loginfo("[Camera] 初始化3个第三视角Zed相机...")
    shared_camera = Camera(camera_type="all", zed_resolution=resolution)

    # 初始化夹爪状态
    try:
        rospy.loginfo("[Setup] 初始化夹爪状态...")
        shared_fa.open_gripper()
        rospy.sleep(1.0)
        rospy.loginfo("[OK] 夹爪初始化完成（打开状态）")
    except Exception as e:
        rospy.logerr(f"夹爪初始化失败: {e}")

    rospy.loginfo("[OK] 共享资源初始化完成\n")

    current_trail = task_idx

    while True:
        try:
            print("\n" + "="*60)
            print(f"📍 准备采集轨迹: trail_{current_trail}")
            print("="*60)
            print("\n🎮 控制说明:")
            print("   [R] 开始/停止录制")
            print("   [G] 切换夹爪")
            print("   [SpaceMouse] 控制机械臂移动")
            print("\n⏸️  等待中... 按 [R] 键开始录制\n")

            collector = CollectDataWithTeleop2(
                task_name=task_name,
                gripper_thres=gripper_thres,
                instruction=instruction,
                trail=current_trail,
                frequency=frequency,
                duration=duration,
                save_interval=save_interval,
                resolution=resolution,
                camera=shared_camera,  # 复用相机
                fa=shared_fa  # 复用机械臂
            )
            collector.run_data_collection(save_dir=data_result_dir)

            # 采集完成后询问用户是否继续
            print("\n")
            print("🎉" + "="*56 + "🎉")
            print(f"✅  trail_{current_trail} 采集完成！共记录 {collector.step_counter} 步数据")
            print("🎉" + "="*56 + "🎉")
            print("\n⏳ 正在保存数据，请稍候...\n")

            # 等待数据保存完成（已在 run_data_collection 中完成）

            print("💾 数据保存完成！\n")
            flush_input_buffer()  # 清空输入缓冲区
            user_input = input("🔄 是否继续采集下一条轨迹? (Y/n，回车默认继续): ").strip().lower()

            if user_input == '' or user_input == 'y' or user_input == 'yes':
                current_trail += 1
                print("\n" + "="*60)
                print(f"📍 下一条轨迹序号: trail_{current_trail}")
                print("="*60)

                # 自动复位机械臂到初始位置并打开夹爪
                print("\n🤖 正在复位机械臂到初始位置...", flush=True)
                try:
                    shared_fa.reset_joints()
                    print("✅ 机械臂已复位到初始位置！", flush=True)
                    print("✋ 正在打开夹爪...", flush=True)
                    shared_fa.open_gripper()
                    print("✅ 夹爪已打开！", flush=True)
                except Exception as e:
                    print(f"⚠️  复位出错: {e}", flush=True)

                print("\n🎮 准备好后按 [R] 键开始录制\n", flush=True)
                # 给用户准备时间
                time.sleep(1)
            else:
                print("\n🏁 采集结束！")
                print(f"📊 本次会话共完成 {current_trail - task_idx + 1} 条轨迹采集")
                print(f"📁 轨迹范围: trail_{task_idx} ~ trail_{current_trail}")
                break

        except KeyboardInterrupt:
            print("\n\n🚨 用户中断，停止采集")
            print(f"📊 已完成 {current_trail - task_idx} 条轨迹采集 (trail_{task_idx} 到 trail_{current_trail - 1})")
            break
        except Exception as e:
            print(f"\n❌ 采集出错: {e}")
            import traceback
            traceback.print_exc()

            # 出错后询问是否继续
            should_continue = False
            while True:
                flush_input_buffer()  # 清空输入缓冲区
                retry_input = input("\n⚠️  是否继续尝试下一条轨迹? (Y继续/N退出): ").strip().lower()
                if retry_input == 'y' or retry_input == 'yes' or retry_input == '':
                    current_trail += 1
                    print(f"\n📍 继续采集 trail_{current_trail}")
                    should_continue = True
                    break
                elif retry_input == 'n' or retry_input == 'no':
                    print("🏁 用户选择退出")
                    should_continue = False
                    break
                else:
                    print("❓ 无效输入，请输入 Y 或 N")

            if should_continue:
                continue
            else:
                break

    # 清理共享资源
    rospy.loginfo("\n[Cleanup] 清理共享资源...")
    try:
        shared_camera.stop()
        rospy.loginfo("[OK] 相机已关闭")
    except Exception as e:
        rospy.logerr(f"关闭相机时出错: {e}")

    rospy.loginfo("[Done] 程序结束")
      
if __name__ == "__main__":
   main()



