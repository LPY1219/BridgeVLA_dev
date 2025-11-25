from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
import numpy as np
import rospy
import pickle as pkl
import cv2


# 修改data_collection_utils.py
def motion2array(translation: np.ndarray, rotation_angles: np.ndarray) -> np.ndarray:
   """
   将运动数据转换为6维数组
   translation: [x, y, z] - 位移增量
   rotation_angles: [rx, ry, rz] - 欧拉角增量
   """
   return np.concatenate([translation, rotation_angles])


def array2motion(array: np.ndarray) -> tuple:
   """
   将6维数组转换回运动数据
   返回: (translation_delta, rotation_angles)
   """
   return array[:3], array[3:]


# 现有的pose函数保持不变，用于其他地方
def pose2array(pose: RigidTransform) -> np.ndarray:
   """转换累积位姿为7维数组（用于当前状态存储）"""
   return np.concatenate([pose.translation, pose.quaternion])


def array2pose(array: np.ndarray) -> RigidTransform:
   """从7维数组重建位姿"""
   return RigidTransform(
       translation=array[:3],
       rotation=RigidTransform.rotation_from_quaternion(array[3:])
   )


def publish_pose(pose:RigidTransform, id:int, timestamp:float, pub:rospy.Publisher, rate:rospy.Rate):
   """
   通过ros发布动作，实现连续操作，其中内涵了sleep
   """
   timestamp = timestamp
   traj_gen_proto_msg = PosePositionSensorMessage(
       id=id, timestamp=timestamp,
       position=pose.translation, quaternion=pose.quaternion
   )
   ros_msg = make_sensor_group_msg(
       trajectory_generator_sensor_msg=sensor_proto2ros_msg(
           traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
   )
   # rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
   pub.publish(ros_msg)
   rate.sleep()
  
def extract_poses_from_file(file_path):
   """
       从保存的连续动作文件中提取出位姿，以及夹爪状态
       Returns:
           pose_traj: List[Dict]: 机械臂动作列表，每个元素是一个Dict，Dict中包含 pose: RigidTransform, Gripper State: Bool
           pose_8d: List[np.ndarray]: 8维动作数组 [x,y,z,qw,qx,qy,qz,gripper]
           pose_len: int: 保存的机械臂动作长度
   """
   traj_dict = pkl.load(open(file_path,'rb'))  # 修复：使用参数而不是字符串
   pose_len = len(traj_dict['current_pose_7d']) # 用pose len动态控制执行时间
   pose_traj = []
   pose_8d = []
   for i, pose_array in enumerate(traj_dict['current_pose_7d']):
       # 获取对应时间点的夹爪状态（修复：先获取再使用）
       gripper_state = traj_dict['keyboard_states'][i]
      
       # 将旋转转换为四元数
       pose = RigidTransform(
           translation=pose_array[:3],
           rotation=RigidTransform.rotation_from_quaternion(pose_array[3:]),
           from_frame = 'franka_tool',
           to_frame= 'world'  # 假设目标坐标系是世界坐标系
       )
      
       # 构造8维动作向量 [位置3D + 四元数4D + 夹爪1D]
       current_action = np.concatenate([
           pose_array,  # 7维位姿
           [gripper_state]  # 1维夹爪状态
       ])
       pose_8d.append(current_action)
      
       pose_with_grip = {
           'pose': pose,
           'gripper_state': gripper_state
       }
       pose_traj.append(pose_with_grip)
   return pose_traj, pose_8d, pose_len




def save_data(result_dict, action, save_dir, idx):
   """一个时刻的数据保存函数"""
   with open(f"{save_dir}/actions/{idx}.pkl", "wb") as f:
       pkl.dump(action, f)
  
   for cam_type, cam_data in result_dict.items():
       if cam_type == "action": continue
      
       # 保存图像数据
       cv2.imwrite(f"{save_dir}/{cam_type}_cam_imgs/{idx}.png", cam_data["rgb"])
      
       # 保存原始数据
       for data_type, values in cam_data.items():
           with open(f"{save_dir}/{cam_type}_cam_{data_type}/{idx}.pkl", "wb") as f:
               pkl.dump(values, f)
              
def save_list_data(picture_list, pose_8d, save_dir):
   """保存图片和动作数据"""
   # 创建必要的目录结构
   import os
   dir_keys = ["3rd_cam_imgs", "wrist_cam_imgs", "3rd_cam_rgb", "3rd_cam_depth", "3rd_cam_pcd",
               "wrist_cam_rgb", "wrist_cam_depth", "wrist_cam_pcd", "actions"]
   for key in dir_keys:
       os.makedirs(os.path.join(save_dir, key), exist_ok=True)
  
   print(f"💾 开始保存数据到: {save_dir}")
   print(f"📊 图片数量: {len(picture_list)}, 动作数量: {len(pose_8d)}")
  
   for idx, (result_dict, action) in enumerate(zip(picture_list, pose_8d)):
       save_data(result_dict, action, save_dir, idx)
       if idx % 10 == 0:  # 每保存10个显示进度
           print(f"📁 已保存: {idx}/{len(picture_list)}")
  
   print(f"✅ 数据保存完成! 共保存 {len(picture_list)} 个样本")
