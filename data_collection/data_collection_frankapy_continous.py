from frankapy import FrankaArm
from autolab_core import RigidTransform
import numpy as np

import time
from utils.real_camera_utils_new import Camera
import cv2
import os
import json
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy

def quat2rotm(quat):
    """Quaternion to rotation matrix.
    quat:w,x,y,z
    """
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    s = w * w + x * x + y * y + z * z
    rotm = np.array(
        [
            [
                1 - 2 * (y * y + z * z) / s,
                2 * (x * y - z * w) / s,
                2 * (x * z + y * w) / s,
            ],
            [
                2 * (x * y + z * w) / s,
                1 - 2 * (x * x + z * z) / s,
                2 * (y * z - x * w) / s,
            ],
            [
                2 * (x * z - y * w) / s,
                2 * (y * z + x * w) / s,
                1 - 2 * (x * x + y * y) / s,
            ],
        ]
    )
    return rotm


def extract_poses_from_file(file_path):
    """
    Reads the JSON content from the specified file and extracts
    all 'pose' arrays from the 'parameterized_poses' field.
    """
    # Read the file and parse the JSON
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Navigate to "parameterized_poses"
    parameterized_poses = data.get("parameter", {}).get("parameterized_poses", [])
    
    # Collect the "pose" array from each item if present
    poses = []
    for item in parameterized_poses:
        pose_data = item.get("pose_with_joint_angles", {}).get("pose", [])
        poses.append(pose_data)
    
    return np.array(poses)

def gripper_open_flag(fa,gripper_thres):
    """
    判断夹爪是否打开
    Returns:
        True: 夹爪打开
        False: 夹爪关闭
    """
    gripper_width=fa.get_gripper_width()
    return gripper_width > gripper_thres

def osc_move(fa, target_pose, gripper_thres, recording=True, recording_continous_frames=True,
            cameras=None, save_dir=None, global_idx=0,save_dir_keypoint="", global_idx_keypoint=0, capture_interval=0.5):
    """修改后的运动控制函数，支持连续数据记录"""
    
    target_pos, target_quat, target_gripper = target_pose
    current_gripper = gripper_open_flag(fa, gripper_thres)  # 初始夹具状态
    
    # 创建目标位姿
    target_pose_fa = RigidTransform(from_frame="franka_tool")
    target_pose_fa.rotation = quat2rotm(target_quat)
    target_pose_fa.translation = target_pos
    
    # 非阻塞运动
    total_time=13
    per_step_time=0.3  # 这个其实没有什么用，因为相机保存频率大概也就1秒3张的样子，所以只有大于0.33s才能发挥作用
    while True:
        try:
            if global_idx_keypoint==0:
                fa.goto_pose(target_pose_fa, use_impedance=False, block=False,duration=1)
            else:
                fa.goto_pose(target_pose_fa, use_impedance=False, block=False,duration=total_time)
            break
        except:
            time.sleep(1)
            continue
    
    # 连续数据记录
    if recording and recording_continous_frames:
        start_time = time.time()
        last_capture = start_time
        
        while True:
            # 获取当前状态
            current_pose = fa.get_pose()
            current_pos = current_pose.translation
            current_quat = current_pose.quaternion
            error = np.linalg.norm(current_pos - target_pos)
            
            # 记录数据
            if time.time() - last_capture >= per_step_time:
                result_dict = cameras.capture()
                
                # 构造动作向量（当前状态+初始夹具状态）
                current_action = np.concatenate([
                    current_pos,
                    current_quat,
                    [current_gripper]  # 使用初始夹具状态
                ])
                
                # 保存数据
                save_data(result_dict, current_action, save_dir, global_idx)
                global_idx += 1
                last_capture = time.time()
            
            # 终止条件
            if error < 0.01:  
                break
            if global_idx_keypoint==0:
                break
            if time.time() - start_time > total_time-per_step_time :
                break
            time.sleep(0.003)
    
    while True:
        try:
            # 确保到达目标
            fa.goto_pose(target_pose_fa, use_impedance=False)
            break
        except:
            time.sleep(1)
    # while True:
    #     if time.time() - start_time < total_time:
    #         time.sleep(1)
    #         continue
    #     else:
    #         break
    # 夹具操作（原有逻辑）
    if current_gripper != target_gripper[0]:
        
        if target_gripper[0]==1:
            fa.open_gripper()
        elif target_gripper[0]==0:
            fa.close_gripper()
        else:
            assert False
        time.sleep(2)
        current_gripper =gripper_open_flag(fa,gripper_thres)
    
    # 单独在另一个路径下记录关键点
    result_dict = cameras.capture()
    target_action = np.concatenate([target_pos, target_quat, target_gripper])
    save_data(result_dict, deepcopy(target_action), save_dir, global_idx)
    result_dict = cameras.capture()
    save_data(result_dict, deepcopy(target_action), save_dir_keypoint, global_idx_keypoint)

    with open(save_dir + f"/instruction.pkl", 'wb') as f:
            pickle.dump(instruction, f)
    with open(save_dir_keypoint + f"/instruction.pkl", 'wb') as f:
            pickle.dump(instruction, f)
    global_idx += 1
    
    return global_idx

def save_data(result_dict, action, save_dir, idx):
    """通用数据保存函数"""
    with open(f"{save_dir}/actions/{idx}.pkl", "wb") as f:
        pickle.dump(action, f)
    
    for cam_type, cam_data in result_dict.items():
        if cam_type == "action": continue
        
        # 保存图像数据
        cv2.imwrite(f"{save_dir}/{cam_type}_cam_imgs/{idx}.png", cam_data["rgb"])
        
        # 保存原始数据
        for data_type, values in cam_data.items():
            with open(f"{save_dir}/{cam_type}_cam_{data_type}/{idx}.pkl", "wb") as f:
                pickle.dump(values, f)



if __name__ == "__main__":

    # # reset_robot
    cameras = Camera(camera_type="3rd")
    time.sleep(2)
    # task_name = "stack_blocks"    # Change 1
    # instruction = "place the green block on the red block" # Change 2
    # task_idx = 29     # Change 3
    # gripper_open =np.array([True,True,False,False,False,True])  # Change 4  True 开 0 关
    # task_name = "place_block_in_shelf"    # Change 1
    # instruction = "place the red block in top shelf" # Change 2
    # task_idx = 9     # Change 3
    # data_result_dir = "/media/casia/data2/lpy/3D_VLA/realworld_datasets"
    # gripper_open =np.array([True,True,False,False,False,True])  # Change 4  True 开 0 关 

    task_name = "debug1"    # Change 1
    instruction = "place the block on the plane" # Change 2
    task_idx = 1     #TODO Change 3
    # data_result_dir = "/media/casia/data2/lpy/3D_VLA/realworld_datasets"
    data_result_dir = "/media/casia/data4/xy/test/6_28"
    gripper_open =np.array([True,True, False, False, False,True])  # Change 4  True 开 0 关 
    expert_action_file = f"/home/casia/Downloads/task.task"  # override_it_everytime
    action_poses = extract_poses_from_file(expert_action_file)
    assert gripper_open.shape[0] == action_poses.shape[0]
    fa = FrankaArm()
    fa.reset_joints()

    save_dir_keypoint = os.path.join(data_result_dir,"keypoint", task_name,str(task_idx))
    os.makedirs(save_dir_keypoint , exist_ok=False)
    save_dir_continous = os.path.join(data_result_dir,"continous", task_name,str(task_idx))
    os.makedirs(save_dir_continous , exist_ok=False)    
    
    dir_keys = ["3rd_cam_imgs", "wrist_cam_imgs", "3rd_cam_rgb", "3rd_cam_depth", "3rd_cam_pcd", \
                "wrist_cam_rgb", "wrist_cam_depth", "wrist_cam_pcd", "actions"]
    for key in dir_keys:
        os.makedirs(os.path.join(save_dir_keypoint, key), exist_ok=True)
        os.makedirs(os.path.join(save_dir_continous, key), exist_ok=True)
    

    # 主程序调用示例
    global_idx = 0
    for idx, action_pose in enumerate(action_poses):

        action_pose = action_pose.reshape(4,4).T  # 为什么要转置

        action_pose = deepcopy(RigidTransform(
        rotation=action_pose[:3,:3],
        translation=action_pose[:3,3]
        ))
        target_pos, target_quat =  action_pose.translation,action_pose.quaternion # 使用frankapy的transform

        target_gripper = gripper_open[idx:idx+1]  # 

        global_idx=osc_move(fa,(target_pos, target_quat, target_gripper),gripper_thres=0.07,recording=True, recording_continous_frames=True, cameras=cameras,
                            global_idx=global_idx,save_dir=save_dir_continous,save_dir_keypoint=save_dir_keypoint, global_idx_keypoint=idx)
        # observations.append(deepcopy(now_obs[0]))
        # Save Results
        
    
    cameras.stop()