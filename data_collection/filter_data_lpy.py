import os
import pickle as pkl
from typing import Dict, List, Union
from PIL import Image
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def load_sample(
   sample_dir: str,
   read_content: bool = True,
   ext_filter: List[str] = None
) -> Dict[str, List]:
   """
   加载样本目录中的所有内容
  
   Args:
       sample_dir: 样本目录路径
       read_content: 是否读取文件内容，False则只返回文件路径
       ext_filter: 文件扩展名过滤器，None表示加载所有文件
  
   Returns:
       字典，key为文件夹名称，value为对应文件夹中内容的列表
   """
   result = {}
  
   # 遍历目录中的所有子文件夹
   for item in os.listdir(sample_dir):
       item_path = os.path.join(sample_dir, item)
      
       # 只处理文件夹
       if os.path.isdir(item_path):
           file_list = []
          
           # 获取文件夹中的所有文件
           for file_name in sorted(os.listdir(item_path)):
               file_path = os.path.join(item_path, file_name)
              
               # 应用扩展名过滤
               if ext_filter is not None:
                   if not any(file_name.endswith(ext) for ext in ext_filter):
                       continue
              
               if read_content:
                   # 根据文件扩展名读取内容
                   if file_name.endswith('.pkl'):
                       try:
                           with open(file_path, 'rb') as f:
                               content = pkl.load(f)
                           file_list.append(content)
                       except ModuleNotFoundError as e:
                           # 处理 numpy 版本兼容性问题
                           print(f"Module error loading {file_path}: {e}")
                           print(f"  Trying with encoding='latin1'...")
                           try:
                               with open(file_path, 'rb') as f:
                                   content = pkl.load(f, encoding='latin1')
                               file_list.append(content)
                           except Exception as e2:
                               print(f"  Still failed: {e2}")
                               file_list.append(None)
                       except Exception as e:
                           print(f"Error loading {file_path}: {e}")
                           print(f"  Error type: {type(e).__name__}")
                           file_list.append(None)
                   elif file_name.endswith('.png'):
                       try:
                           image = Image.open(file_path)
                           file_list.append(np.array(image))
                       except Exception as e:
                           print(f"Error loading {file_path}: {e}")
                           file_list.append(None)
                   elif file_name.endswith('.txt'):
                       try:
                           with open(file_path, 'r', encoding='utf-8') as f:
                               content = f.read().strip()
                           file_list.append(content)
                       except Exception as e:
                           print(f"Error loading {file_path}: {e}")
                           file_list.append(None)
                   else:
                       # 其他文件类型，尝试作为文本读取
                       try:
                           with open(file_path, 'r', encoding='utf-8') as f:
                               content = f.read()
                           file_list.append(content)
                       except:
                           file_list.append(file_path)
               else:
                   # 不读取内容，只返回文件路径
                   file_list.append(file_path)
                  

           result[item] = file_list

   # 读取指令文件
   with open(os.path.join(sample_dir, "instruction.txt"), 'r', encoding='utf-8') as f:
       instruction = f.read().strip()
   result['instruction'] = instruction

   # 读取外参矩阵
   # 新格式: extrinsics.pkl (字典，包含3个相机外参)
   extrinsics_path = os.path.join(sample_dir, "extrinsics.pkl")
   if os.path.exists(extrinsics_path):
       try:
           with open(extrinsics_path, 'rb') as f:
               extrinsics = pkl.load(f)
           result['extrinsics'] = extrinsics
           print(f"已加载 extrinsics.pkl (新格式，包含 {len(extrinsics)} 个相机外参)")
       except Exception as e:
           print(f"加载 extrinsics.pkl 失败: {e}")

   # 旧格式: extrinic.pkl (单一外参矩阵)
   extrinic_path = os.path.join(sample_dir, "extrinic.pkl")
   if os.path.exists(extrinic_path):
       try:
           with open(extrinic_path, 'rb') as f:
               extrinic = pkl.load(f)
           result['extrinic'] = extrinic
           print(f"已加载 extrinic.pkl (旧格式)")
       except Exception as e:
           print(f"加载 extrinic.pkl 失败: {e}")

   return result


def quaternion_angle_difference_scipy(q1, q2):
    """使用scipy计算四元数角度差异（弧度）
    
    Args:
        q1, q2: 四元数，格式为[w, x, y, z]
    
    Returns:
        两个四元数之间的角度差异（弧度）
    """
    # 注意：scipy期望[x,y,z,w]格式，您的数据是[w,x,y,z]
    # 转换格式
    q1_scipy = [q1[1], q1[2], q1[3], q1[0]]  # wxyz -> xyzw
    q2_scipy = [q2[1], q2[2], q2[3], q2[0]]  # wxyz -> xyzw
    
    try:
        r1 = R.from_quat(q1_scipy)
        r2 = R.from_quat(q2_scipy)
        
        # 计算相对旋转
        relative_rotation = r2 * r1.inv()
        
        # 获取旋转角度（弧度）
        angle = relative_rotation.magnitude()
        
        return angle
    except Exception as e:
        print(f"Error calculating quaternion difference: {e}")
        return 0.0


def filter_data(data_path: str, thres_xyz=0.01, thres_rotation_deg=1.0):
    """
    使用累积阈值进行数据过滤
    从保留的数据点开始，向后遍历直到找到满足条件的下一个数据点
    
    Args:
        data_path: 数据路径
        thres_xyz: 位置变化阈值（米）
        thres_rotation_deg: 旋转角度阈值（度）
    
    Returns:
        过滤后的数据字典
    """
    data = load_sample(data_path)
    poses = data['poses']
    gripper_states = data['gripper_states']
    print(f"过滤前时间步数量：{len(poses)}")
    
    # 将角度阈值转换为弧度
    thres_rotation_rad = np.deg2rad(thres_rotation_deg)
    
    keep_indices = [0]  # 始终保留第一个数据点
    last_kept_index = 0  # 上一个保留的数据点索引
    
    i = 1
    while i < len(poses):
        pose_reference = poses[last_kept_index]  # 参考点（上一个保留的点）
        pose_curr = poses[i]
        
        # 计算与参考点的位置差异
        xyz_diff = np.linalg.norm(pose_curr[:3] - pose_reference[:3])
        
        # 计算与参考点的四元数差异
        quat_reference = pose_reference[3:]  # [w, x, y, z]
        quat_curr = pose_curr[3:]  # [w, x, y, z]
        rotation_diff = quaternion_angle_difference_scipy(quat_reference, quat_curr)
        
        # 计算与参考点的夹爪状态差异
        gripper_diff = abs(int(gripper_states[i]) - int(gripper_states[last_kept_index]))
        
        # 检查是否满足保留条件
        if (xyz_diff >= thres_xyz) or (rotation_diff >= thres_rotation_rad) or (gripper_diff != 0):
            keep_indices.append(i)
            last_kept_index = i  # 更新参考点
            print(f"保留数据点 {i}: 位置变化={xyz_diff:.4f}m, 角度变化={np.rad2deg(rotation_diff):.2f}°, 夹爪变化={gripper_diff}")
        
        i += 1

    # 需要排除的键：instruction（字符串）、extrinsics（外参字典）、extrinic（外参矩阵）
    exclude_keys = {'instruction', 'extrinsics', 'extrinic'}

    # 只对时序数据（列表）进行索引过滤
    filtered_data = {k: [v[i] for i in keep_indices] for k, v in data.items() if k not in exclude_keys}

    # 单独添加不需要过滤的数据
    filtered_data['instruction'] = data['instruction']

    # 保留外参矩阵（如果存在）
    if 'extrinsics' in data:
        filtered_data['extrinsics'] = data['extrinsics']
    if 'extrinic' in data:
        filtered_data['extrinic'] = data['extrinic']

    print(f"过滤后时间步数量：{len(filtered_data['poses'])}")
    print(f"过滤掉的数据点: {len(poses) - len(filtered_data['poses'])}")
    print(f"保留比例: {len(filtered_data['poses'])/len(poses)*100:.2f}%")
    return filtered_data

      
def save_collected_data(save_dir: str, data_dict, trail_id: int):
    """
    保存过滤后的数据，支持新的3相机格式：
    - 新格式（3个第三视角相机）：
        '3rd_1_bgr_images', '3rd_1_bgr', '3rd_1_depth', '3rd_1_pcd'
        '3rd_2_bgr_images', '3rd_2_bgr', '3rd_2_depth', '3rd_2_pcd'
        '3rd_3_bgr_images', '3rd_3_bgr', '3rd_3_depth', '3rd_3_pcd'
        'extrinsics' (字典，包含3个相机的外参)
    - 旧格式兼容：
        '3rd_bgr_images', '3rd_bgr', 'wrist_bgr_images', 'wrist_bgr'
        'bgr_images', 'depth', 'pcd', 'extrinic' (单一外参矩阵)
    - 在写 PNG 前，将 RGB/RGBA 转为 BGR/BGRA，避免颜色颠倒
    - 其他目录逐帧写 PKL
    """
    import os
    import pickle as pkl
    import numpy as np
    import cv2

    out_trail_dir = os.path.join(save_dir, f"trail_{trail_id}")
    os.makedirs(out_trail_dir, exist_ok=False)

    # 以 poses 的长度为准
    actual_length = len(data_dict['poses'])

    # 写指令（若存在且为字符串）
    instr = data_dict.get('instruction', None)
    if isinstance(instr, str):
        with open(os.path.join(out_trail_dir, "instruction.txt"), 'w', encoding='utf-8') as f:
            f.write(instr)

    # 保存外参矩阵
    # 新格式: extrinsics (字典)
    if 'extrinsics' in data_dict:
        with open(os.path.join(out_trail_dir, "extrinsics.pkl"), 'wb') as f:
            pkl.dump(data_dict['extrinsics'], f, protocol=pkl.HIGHEST_PROTOCOL)
        print("已保存 extrinsics.pkl (新格式，3个相机外参)")
    # 旧格式: extrinic (单一矩阵)
    elif 'extrinic' in data_dict:
        with open(os.path.join(out_trail_dir, "extrinic.pkl"), 'wb') as f:
            pkl.dump(data_dict['extrinic'], f, protocol=pkl.HIGHEST_PROTOCOL)
        print("已保存 extrinic.pkl (旧格式)")

    # 处理的目录（存在才写，不存在跳过）
    dir_names = [
        # 新格式：3个第三视角相机
        '3rd_1_bgr_images', '3rd_1_bgr', '3rd_1_depth', '3rd_1_pcd',
        '3rd_2_bgr_images', '3rd_2_bgr', '3rd_2_depth', '3rd_2_pcd',
        '3rd_3_bgr_images', '3rd_3_bgr', '3rd_3_depth', '3rd_3_pcd',
        # 旧格式：1个第三视角 + 1个腕部相机
        '3rd_bgr_images', '3rd_bgr',
        'wrist_bgr_images', 'wrist_bgr',
        'bgr_images',
        'depth', 'pcd',
        # 公共数据
        'poses', 'gripper_states', 'joints'
    ]

    def _write_png(img_path: str, img):
        """在写 PNG 前做最小必要的通道/类型处理"""
        img_to_save = img
        if isinstance(img, np.ndarray):
            # 三/四通道：RGB(A) -> BGR(A)
            if img.ndim == 3:
                if img.shape[2] == 3:
                    img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif img.shape[2] == 4:
                    img_to_save = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            # 确保 uint8
            if img_to_save.dtype != np.uint8:
                img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
        cv2.imwrite(img_path, img_to_save)

    for dir_name in dir_names:
        if dir_name not in data_dict:
            continue

        dir_path = os.path.join(out_trail_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)

        # 图片目录：逐帧写 PNG（做 RGB->BGR 转换）
        if 'bgr_images' in dir_name:
            print(f"正在保存 {dir_name} 图像文件")
            seq = data_dict[dir_name]
            for i in range(actual_length):
                img_path = os.path.join(dir_path, f"{i:06d}.png")
                _write_png(img_path, seq[i])

        # 其余目录：逐帧写 PKL
        else:
            print(f"正在保存 {dir_name} 数组")
            seq = data_dict[dir_name]
            for i in range(actual_length):
                with open(os.path.join(dir_path, f"{i:06d}.pkl"), 'wb') as f:
                    pkl.dump(seq[i], f, protocol=pkl.HIGHEST_PROTOCOL)





def batch_filter_data(source_dir: str, target_dir: str, thres_xyz=0.01, thres_rotation_deg=1.0):
    """
    批量过滤多个trail数据
    
    Args:
        source_dir: 源文件夹路径，包含多个trail_xx子文件夹
        target_dir: 目标文件夹路径，用于保存过滤后的数据
        thres_xyz: 位置变化阈值（米）
        thres_rotation_deg: 旋转角度阈值（度）
    """
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有trail文件夹
    trail_folders = []
    for item in os.listdir(source_dir):
        if item.startswith('trail_') and os.path.isdir(os.path.join(source_dir, item)):
            trail_folders.append(item)
    
    # 按trail编号排序
    trail_folders.sort(key=lambda x: int(x.split('_')[1]))
    
    print(f"发现 {len(trail_folders)} 个trail文件夹: {trail_folders}")
    
    # 批量处理每个trail
    success_count = 0
    failed_trails = []
    
    for trail_folder in trail_folders:
        try:
            # 提取trail编号
            trail_id = int(trail_folder.split('_')[1])
            trail_path = os.path.join(source_dir, trail_folder)
            
            print(f"\n{'='*50}")
            print(f"正在处理 {trail_folder} (ID: {trail_id})")
            print(f"{'='*50}")
            
            # 检查目标文件夹是否已存在
            target_trail_path = os.path.join(target_dir, trail_folder)
            if os.path.exists(target_trail_path):
                print(f"警告: {trail_folder} 已存在于目标文件夹，跳过...")
                continue
            
            # 过滤数据
            filtered_data = filter_data(trail_path, thres_xyz=thres_xyz, thres_rotation_deg=thres_rotation_deg)
            
            # 保存过滤后的数据
            save_collected_data(save_dir=target_dir, data_dict=filtered_data, trail_id=trail_id)
            
            print(f"✅ {trail_folder} 处理完成")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 处理 {trail_folder} 时出错: {e}")
            failed_trails.append(trail_folder)
            continue
    
    print(f"\n{'='*50}")
    print(f"批量处理完成!")
    print(f"成功处理: {success_count}/{len(trail_folders)} 个trail")
    if failed_trails:
        print(f"处理失败的trail: {failed_trails}")
    print(f"{'='*50}")


# 示例使用
if __name__ == "__main__":
    # 单个trail处理示例
    # filtered_data = filter_data("/media/casia/data4/lpy/3zed_data/raw_data/put_lion_on_top_shelf/trail_1",
    #                            thres_xyz=0.01,
    #                            thres_rotation_deg=1.0)
    # save_collected_data(save_dir="/media/casia/data4/lpy/3zed_data/filtered_data/put_lion_on_top_shelf",
    #                    data_dict=filtered_data,
    #                    trail_id=1)

    # 批量处理示例 - 新格式（3个第三视角相机）
    batch_filter_data(
        source_dir="/media/casia/data4/lpy/3zed_data/raw_data/put_lion_on_top_shelf",  # 新格式：3个第三视角相机
        target_dir="/media/casia/data4/lpy/3zed_data/filter_data/put_lion_on_top_shelf",
        thres_xyz=0.01,
        thres_rotation_deg=3.0
    )

    # 批量处理示例 - 旧格式（1个第三视角 + 1个腕部相机）
    # batch_filter_data(
    #     source_dir="/media/casia/data4/wxn/data/DP0/put_coke_can_on_the_upper_drawer",  # 旧格式
    #     target_dir="/media/casia/T7/Franka_data/put_coke_can_on_the_upper_drawer",
    #     thres_xyz=0.01,
    #     thres_rotation_deg=2.0
    # )