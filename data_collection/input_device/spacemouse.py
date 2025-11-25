"""
Franka SpaceMouse接口

基于共享内存的高效SpaceMouse输入处理，专门针对Franka机器人优化。
"""

import multiprocessing as mp
import numpy as np
import time
from typing import Tuple, Optional

try:
    import pyspacemouse
except ImportError:
    print("Warning: pyspacemouse not installed. Please install: pip install pyspacemouse")
    pyspacemouse = None
import os, sys

# 添加当前目录和子目录到Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class FrankaSpacemouse(mp.Process):
    """
    Franka机器人专用SpaceMouse接口
    
    针对Franka的工作特性和控制需求进行了优化，提供6DOF输入处理
    和智能按钮模式切换。
    
    Features:
        - 高频数据采集（200Hz）
        - 死区滤波减少噪声
        - 坐标系转换适配Franka
        - 按钮模式智能切换
        - 共享内存高效通信
        
    Examples:
        >>> from multiprocessing.managers import SharedMemoryManager
        >>> 
        >>> with SharedMemoryManager() as shm_manager:
        ...     spacemouse = FrankaSpacemouse(
        ...         shm_manager=shm_manager,
        ...         deadzone=0.02,
        ...         position_sensitivity=1.5
        ...     )
        ...     
        ...     with spacemouse:
        ...         motion = spacemouse.get_motion_state()
        ...         print(f"运动状态: {motion}")
        ...         
        ...         if spacemouse.is_rotation_enabled():
        ...             print("当前为旋转控制模式")
    """

    def __init__(self, 
                 shm_manager, 
                 get_max_k=30, 
                 frequency=200,
                 max_value=500, 
                 deadzone=(0.02, 0.02, 0.02, 0.02, 0.02, 0.02), 
                 dtype=np.float32,
                 n_buttons=2,
                 # Franka特定参数
                 coordinate_frame='world',  # 'world' or 'tool'
                 position_sensitivity=1.0,
                 rotation_sensitivity=1.0,
                 button_functions=None,
                 debug=False  # 调试模式开关
                 ):
        """
        初始化Franka SpaceMouse控制器
        
        Args:
            shm_manager: 共享内存管理器
            get_max_k: 环形缓冲区最大容量
            frequency: 数据更新频率 (Hz)
            max_value: SpaceMouse最大输入值 (300无线版本, 500有线版本)
            deadzone: 死区设置，6个轴的死区阈值 [0,1]
            dtype: 数据类型
            n_buttons: 按钮数量
            coordinate_frame: 坐标系 ('world': 世界坐标系, 'tool': 工具坐标系)
            position_sensitivity: 位置灵敏度倍数
            rotation_sensitivity: 旋转灵敏度倍数
            button_functions: 自定义按钮功能映射
        """
        super().__init__()
        
        # 死区处理
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        # 存储参数
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons
        self.coordinate_frame = coordinate_frame
        self.position_sensitivity = position_sensitivity
        self.rotation_sensitivity = rotation_sensitivity
        self.debug = debug
        
        # 按钮功能映射
        if button_functions is None:
            self.button_functions = {
                0: 'rotation_mode',    # 左键：启用旋转控制
                1: 'z_unlock'          # 右键：解锁Z轴
            }
        else:
            self.button_functions = button_functions

        # Franka坐标系转换矩阵
        # SpaceMouse的原始坐标系到机器人世界坐标系的转换
        self.tx_spacemouse_to_world = np.array([
            [0, 1, 0],    # SpaceMouse Z轴 -> 世界 -X轴
            [1, 0, 0],     # SpaceMouse X轴 -> 世界 Y轴  
            [0, 0, 1]      # SpaceMouse Y轴 -> 世界 Z轴
        ], dtype=dtype)

        # 构建共享内存数据结构
        example_data = {
            # 6DOF运动数据: [x,y,z,rx,ry,rz]
            'motion_state': np.zeros((6,), dtype=dtype),
            'motion_state_raw': np.zeros((6,), dtype=dtype),
            # 按钮状态
            'button_state': np.zeros((n_buttons,), dtype=bool),
            # 控制模式状态 - 转换为numpy数组
            'rotation_enabled': False,
            'z_unlocked': False,
            'coordinate_frame_index': 0,  # 0: world, 1: tool
            # 时间戳
            'receive_timestamp': time.time()
        }
        
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example_data,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # 进程同步
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

    # ======= 状态获取API ==========
    def get_motion_state(self, apply_sensitivity=True) -> np.ndarray:
        """
        获取处理后的运动状态
        
        Args:
            apply_sensitivity: 是否应用灵敏度设置
            
        Returns:
            6维运动状态 [x,y,z,rx,ry,rz]，范围 [-1,1]
        """
        data = self.ring_buffer.get()
        motion_state = data['motion_state'].copy()
        
        if apply_sensitivity:
            motion_state[:3] *= self.position_sensitivity
            motion_state[3:] *= self.rotation_sensitivity
            
        return motion_state
    
    def get_motion_state_raw(self, apply_sensitivity = True) -> np.ndarray:
        """获取原始运动状态（未经坐标转换）"""
        data = self.ring_buffer.get()
        motion_state = data['motion_state_raw'].copy()

        if apply_sensitivity:
            motion_state[:3] *= self.position_sensitivity
            motion_state[3:] *= self.rotation_sensitivity
        return motion_state
    
    def get_control_mode(self) -> dict:
        """获取当前控制模式"""
        data = self.ring_buffer.get()
        coordinate_frames = ['world', 'tool']
        return {
            'rotation_enabled': bool(data['rotation_enabled']),
            'z_unlocked': bool(data['z_unlocked']),
            'coordinate_frame': coordinate_frames[int(data['coordinate_frame_index'])]
        }
    
    def get_button_state(self) -> np.ndarray:
        """获取按钮状态"""
        data = self.ring_buffer.get()
        return data['button_state']
    
    def is_button_pressed(self, button_id: int) -> bool:
        """检查特定按钮是否被按下"""
        return self.get_button_state()[button_id]
    
    def is_rotation_enabled(self) -> bool:
        """检查旋转控制是否启用"""
        return self.get_control_mode()['rotation_enabled']
    
    def is_z_unlocked(self) -> bool:
        """检查Z轴是否解锁"""
        return self.get_control_mode()['z_unlocked']

    # ======= 坐标转换方法 ==========
    def transform_to_world_frame(self, motion_raw: np.ndarray) -> np.ndarray:
        """将SpaceMouse原始输入转换到世界坐标系"""
        motion_world = np.zeros_like(motion_raw)
        motion_world[:3] = self.tx_spacemouse_to_world @ motion_raw[:3]
        motion_world[3:] = self.tx_spacemouse_to_world @ motion_raw[3:]
        return motion_world

    def apply_control_logic(self, motion_state: np.ndarray, 
                           button_state: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        应用控制逻辑（按钮模式切换等）
        
        Args:
            motion_state: 基础运动状态
            button_state: 按钮状态
            
        Returns:
            (处理后的运动状态, 控制模式信息字典)
        """
        processed_motion = motion_state.copy()
        
        rotation_enabled = button_state[0] if 0 < len(button_state) else False
        z_unlocked = button_state[1] if 1 < len(button_state) else True
        coordinate_frame_index = 0 if self.coordinate_frame == 'world' else 1
        
        # 按钮逻辑：根据按钮状态修改运动命令
        if not rotation_enabled:
            # 仅平移模式：禁用旋转
            processed_motion[3:] = 0
        else:
            # 仅旋转模式：禁用平移
            processed_motion[:3] = 0
            
        if not z_unlocked:
            # Z轴锁定：禁用Z轴运动
            processed_motion[2] = 0
        
        # 返回处理后的数据，包含展平的控制模式信息
        control_data = {
            'rotation_enabled': rotation_enabled,
            'z_unlocked': z_unlocked,
            'coordinate_frame_index': coordinate_frame_index
        }
            
        return processed_motion, control_data

    #========== 启动停止API ===========
    def start(self, wait=True):
        """启动SpaceMouse进程"""
        super().start()
        if wait:
            self.ready_event.wait()
    
    def stop(self, wait=True):
        """停止SpaceMouse进程"""
        self.stop_event.set()
        if wait:
            self.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= 主循环 ==========
    def run(self):
        """主运行循环"""
        if pyspacemouse is None:
            print("Error: pyspacemouse未安装，无法使用SpaceMouse")
            return
            
        success = pyspacemouse.open()
        if not success:
            print("Error: 无法打开SpaceMouse设备")
            return
            
        try:
            # 初始化状态
            motion_raw = np.zeros((6,), dtype=self.dtype)
            motion_state = np.zeros((6,), dtype=self.dtype)
            button_state = np.zeros((self.n_buttons,), dtype=bool)
            
            coordinate_frame_index = 0 if self.coordinate_frame == 'world' else 1

            # 发送初始状态
            self.ring_buffer.put({
                'motion_state': motion_state,
                'motion_state_raw': motion_raw,
                'button_state': button_state,
                'rotation_enabled': False,
                'z_unlocked': False,
                'coordinate_frame_index': coordinate_frame_index,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()

            print(f"[FrankaSpacemouse] SpaceMouse控制器已启动")
            print(f"  - 左键: {self.button_functions.get(0, 'undefined')}")
            print(f"  - 右键: {self.button_functions.get(1, 'undefined')}")
            print(f"  - 坐标系: {self.coordinate_frame}")

            while not self.stop_event.is_set():
                state = pyspacemouse.read()
                receive_timestamp = time.time()
                
                if state is not None:
                    # 调试：检查state对象的所有属性（仅在调试模式下）
                    if self.debug and not hasattr(self, '_debug_printed'):
                        print(f"[Debug] State对象类型: {type(state)}")
                        print(f"[Debug] State对象属性: {dir(state)}")
                        if hasattr(state, '__dict__'):
                            print(f"[Debug] State对象内容: {vars(state)}")
                        self._debug_printed = True
                    
                    # pyspacemouse的标准API可能是不同的属性名
                    # 尝试多种可能的属性名
                    try:
                        # 方法1：标准属性名
                        if hasattr(state, 'x'):
                            raw_values = [state.x, state.y, state.z, state.roll, state.pitch, state.yaw]
                        # 方法2：可能的其他属性名
                        elif hasattr(state, 'translation') and hasattr(state, 'rotation'):
                            raw_values = list(state.translation) + list(state.rotation)
                        # 方法3：可能的数组形式
                        elif hasattr(state, 'axis'):
                            raw_values = list(state.axis)
                        else:
                            print(f"[Warning] 未知的state格式: {state}")
                            raw_values = [0, 0, 0, 0, 0, 0]
                        
                        # 检查是否有非零值（仅在调试模式下）
                        if self.debug and any(abs(val) > 0 for val in raw_values):
                            print(f"[Debug] 原始值: {raw_values}")
                        
                        # 处理运动事件 - 注意：pyspacemouse的值已经是标准化的[-1,1]范围
                        # 不需要再除以max_value！
                        motion_raw[:] = np.array(raw_values[:6], dtype=self.dtype)
                        
                        # 如果有运动，显示处理后的结果（仅在调试模式下）
                        if self.debug:
                            if np.any(np.abs(motion_raw) > 0.001):
                                print(f"[Debug] 处理后motion_raw: {motion_raw}")
                            elif np.any(np.abs(motion_raw) > 0):
                                print(f"[Debug] motion_raw太小被忽略: {motion_raw}")
                            
                    except Exception as e:
                        print(f"[Error] 处理state数据时出错: {e}")
                        continue
                    
                    # 应用死区
                    motion_before_deadzone = motion_raw.copy()
                    is_dead = (-self.deadzone < motion_raw) & (motion_raw < self.deadzone)
                    motion_raw[is_dead] = 0
                    
                    # 显示死区过滤的效果（仅在调试模式下）
                    if self.debug and np.any(np.abs(motion_before_deadzone) > 0.001):
                        print(f"[Debug] 死区过滤前: {motion_before_deadzone}")
                        print(f"[Debug] 死区阈值: {self.deadzone}")
                        print(f"[Debug] 死区过滤后: {motion_raw}")
                    
                    # 坐标转换
                    if self.coordinate_frame == 'world':
                        motion_state = self.transform_to_world_frame(motion_raw)
                        if self.debug and np.any(np.abs(motion_raw) > 0.001):
                            print(f"[Debug] 坐标转换前: {motion_raw}")
                            print(f"[Debug] 坐标转换后: {motion_state}")
                    else:
                        motion_state = motion_raw.copy()
                    
                    # 处理按钮状态 - pyspacemouse的buttons
                    # 根据pyspacemouse的实际API，buttons通常有left和right属性
                    try:
                        if hasattr(state.buttons, 'left') and hasattr(state.buttons, 'right'):
                            # pyspacemouse的ButtonState对象有left和right属性
                            if self.n_buttons >= 1:
                                button_state[0] = bool(state.buttons.left)
                            if self.n_buttons >= 2:
                                button_state[1] = bool(state.buttons.right)
                        elif hasattr(state.buttons, '__iter__'):
                            # 如果buttons是可迭代的（列表或元组）
                            for i in range(min(self.n_buttons, len(state.buttons))):
                                button_state[i] = bool(state.buttons[i])
                        else:
                            # 尝试按位操作
                            buttons_value = int(state.buttons)
                            for i in range(min(self.n_buttons, 8)):
                                button_state[i] = bool(buttons_value & (1 << i))
                    except (ValueError, TypeError, AttributeError) as e:
                        # 如果处理失败，保持当前状态并输出调试信息
                        if hasattr(state.buttons, '__dict__'):
                            print(f"[Debug] ButtonState属性: {vars(state.buttons)}")
                        else:
                            print(f"[Debug] ButtonState类型: {type(state.buttons)}, 值: {state.buttons}")
                        pass
                
                # 应用控制逻辑并发送状态
                motion_processed, control_data = self.apply_control_logic(
                    motion_state, button_state)
                
                self.ring_buffer.put({
                    'motion_state': motion_processed,
                    'motion_state_raw': motion_raw,
                    'button_state': button_state.copy(),
                    'rotation_enabled': control_data['rotation_enabled'],
                    'z_unlocked': control_data['z_unlocked'],
                    'coordinate_frame_index': control_data['coordinate_frame_index'],
                    'receive_timestamp': receive_timestamp
                })
                
                time.sleep(1/self.frequency)
                    
        except Exception as e:
            print(f"[FrankaSpacemouse] 运行错误: {e}")
        finally:
            pyspacemouse.close()
            print("[FrankaSpacemouse] SpaceMouse控制器已关闭")


def test_franka_spacemouse(debug=True):
    """测试Franka SpaceMouse接口
    
    Args:
        debug: 是否启用调试模式
    """
    from multiprocessing.managers import SharedMemoryManager
    
    with SharedMemoryManager() as shm_manager:
        with FrankaSpacemouse(
            shm_manager=shm_manager,
            deadzone=0.005,  # 降低死区阈值，从0.05改为0.005
            position_sensitivity=1.5,
            rotation_sensitivity=1.0,
            debug=debug  # 调试模式可控制
        ) as sm:
            print("测试FrankaSpaceMouse - 按Ctrl+C停止")
            try:
                for i in range(1000):
                    motion_state = sm.get_motion_state()
                    control_mode = sm.get_control_mode()
                    
                    motion_norm = np.linalg.norm(motion_state)
                    print(f"\r运动状态: [{motion_state[0]:.3f}, {motion_state[1]:.3f}, {motion_state[2]:.3f}, "
                          f"{motion_state[3]:.3f}, {motion_state[4]:.3f}, {motion_state[5]:.3f}] "
                          f"强度: {motion_norm:.3f}, "
                          f"旋转模式: {control_mode['rotation_enabled']}, "
                          f"Z解锁: {control_mode['z_unlocked']}", 
                          end='', flush=True)
                    
                    time.sleep(1/50)  # 50Hz显示更新
            except KeyboardInterrupt:
                print("\n测试结束")


if __name__ == '__main__':
    import sys
    # 检查命令行参数来控制调试模式
    debug_mode = '--debug' in sys.argv or '-d' in sys.argv
    test_franka_spacemouse(debug=debug_mode)
