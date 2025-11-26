import pickle as pkl
import numpy as np


from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from scipy.spatial.transform import Rotation as R


import rospy
import os

def gripper_open_flag(fa,gripper_thres):
    gripper_width=fa.get_gripper_width()
    return gripper_width > gripper_thres



def load_trajectory_from_teleop_data(data_dir):
    """从遥操作数据目录加载轨迹"""
    print(f"Loading trajectory from: {data_dir}")
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # 获取数据长度
    poses_dir = os.path.join(data_dir, "poses")
    if not os.path.exists(poses_dir):
        raise FileNotFoundError(f"Poses directory not found: {poses_dir}")
    
    pkl_files = [f for f in os.listdir(poses_dir) if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {poses_dir}")
    
    data_length = len(pkl_files)
    print(f"Found {data_length} trajectory points")
    
    # 加载轨迹数据
    trajectory = []
    
    for i in range(data_length):
        try:
            # 加载位姿 [x, y, z, qw, qx, qy, qz]
            pose_path = os.path.join(data_dir, "poses", f"{i:06d}.pkl")
            with open(pose_path, 'rb') as f:
                pose_7d = pkl.load(f)
            
            # 加载夹爪状态
            gripper_path = os.path.join(data_dir, "gripper_states", f"{i:06d}.pkl")
            with open(gripper_path, 'rb') as f:
                gripper_state = pkl.load(f)
            
            # 确保pose_7d格式正确
            if len(pose_7d) != 7:
                raise ValueError(f"Invalid pose format at index {i}: expected 7 values, got {len(pose_7d)}")
            
            # 转换为RigidTransform
            pose = RigidTransform(
                translation=pose_7d[:3],
                rotation=RigidTransform.rotation_from_quaternion(pose_7d[3:]),
                from_frame='franka_tool',
                to_frame='world'
            )
            
            trajectory.append({
                'pose': pose,
                'gripper_state': gripper_state
            })
            
        except Exception as e:
            rospy.logerr(f"Error loading data at index {i}: {e}")
            raise
    
    return trajectory


if __name__ == "__main__":
    # 配置参数
    DATA_DIR = "/media/casia/data4/wxn/data/final_data2/p&p/trail_19"  # 修改为你的数据目录
    # 初始化
    fa = FrankaArm()
    fa.reset_joints()
    
    rospy.loginfo('Loading trajectory from teleop data...')
    
    # 加载轨迹数据
    try:
        pose_traj = load_trajectory_from_teleop_data(DATA_DIR)
    except Exception as e:
        rospy.logerr(f"Failed to load trajectory: {e}")
        exit(1)
    
    pose_len = len(pose_traj)
    rospy.loginfo(f'Loaded trajectory with {pose_len} points')
    
    if pose_len < 2:
        rospy.logerr("Trajectory too short, need at least 2 points")
        exit(1)
    
    # 计算回放参数
    dt = 0.1
    T = dt * pose_len
    
    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
    rate = rospy.Rate(1 / dt)
    
    
    
    # 设置初始夹爪状态
    initial_gripper = pose_traj[0]['gripper_state']
    try:
        if initial_gripper:
            fa.close_gripper()
        else:
            fa.open_gripper()
        rospy.sleep(1.0)
    except Exception as e:
        rospy.logwarn(f"Failed to set initial gripper state: {e}")
    
    rospy.loginfo('Starting trajectory replay...')
    
    fa.goto_pose(pose_traj[1]['pose'], duration=T, dynamic=True,buffer_time=10)
    
    init_time = rospy.Time.now().to_time()
    previous_gripper_state = initial_gripper
    
    
    for i in range(2, len(pose_traj)):
        timestamp = rospy.Time.now().to_time() - init_time
        pose1 = fa.get_pose()
        current_pose = pose_traj[i]['pose']
        current_gripper = pose_traj[i]['gripper_state']
        # 发布位姿命令
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, 
            timestamp=timestamp,
            position=current_pose.translation,  # 确保是list格式
            quaternion=current_pose.quaternion  # 确保是list格式
        )
            
        # 创建ROS消息并设置header
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg,SensorDataMessageType.POSE_POSITION
            )
        )
        # rospy.loginfo("Publishing: ID {} - Gripper {}".format(i, 'CLOSED' if current_gripper else "OPEN"))

        
        pub.publish(ros_msg)
        pose2 = fa.get_pose()
        rospy.loginfo(f"目标与当前状态之差: {np.linalg.norm(current_pose.translation - pose1.translation)} m")
        rospy.loginfo(f"两个目标状态之差: {np.linalg.norm(current_pose.translation - pose_traj[i-1]['pose'].translation)} m")
        # 处理夹爪状态变化
        if previous_gripper_state is None or current_gripper != previous_gripper_state:
            rospy.loginfo("Changing gripper state to: {}".format("CLOSED" if current_gripper else "OPEN"))

            if current_gripper:
                fa.close_gripper()
            else:
                fa.open_gripper()
            rospy.sleep(1.0)
        previous_gripper_state = current_gripper
        rate.sleep()
            
        
        # rospy.loginfo("Trajectory replay completed successfully!")
        
    term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
    pub.publish(ros_msg)


    rospy.loginfo('Done')