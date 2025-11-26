"""
共享内存环形缓冲区实现

提供无锁FILO共享内存数据结构，用于存储时间序列数据。
"""

from typing import Dict, List, Union
from queue import Empty
import numbers
import time
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from .shared_ndarray import SharedNDArray
from .shared_memory_util import ArraySpec, SharedAtomicCounter


class SharedMemoryRingBuffer:
    """
    无锁FILO共享内存数据结构
    存储numpy数组字典的时间序列
    
    专为高频数据流设计，支持时间戳管理和自动速率控制。
    特别适用于传感器数据、控制命令等实时数据流。
    
    Examples:
        >>> from multiprocessing.managers import SharedMemoryManager
        >>> import numpy as np
        >>> import time
        >>> 
        >>> with SharedMemoryManager() as shm_manager:
        ...     examples = {
        ...         'position': np.array([0.0, 0.0, 0.0]),
        ...         'timestamp': time.time()
        ...     }
        ...     buffer = SharedMemoryRingBuffer.create_from_examples(
        ...         shm_manager, examples, get_max_k=10, put_desired_frequency=100)
        ...     
        ...     # 放入数据
        ...     data = {'position': np.array([1.0, 2.0, 3.0]), 'timestamp': time.time()}
        ...     buffer.put(data)
        ...     
        ...     # 获取最新数据
        ...     latest = buffer.get()
        ...     print(latest['position'])  # [1. 2. 3.]
    """

    def __init__(self, 
                 shm_manager: SharedMemoryManager,
                 array_specs: List[ArraySpec],
                 get_max_k: int,
                 get_time_budget: float,
                 put_desired_frequency: float,
                 safety_margin: float = 1.5
                 ):
        """
        初始化共享内存环形缓冲区
        
        Args:
            shm_manager: 管理共享内存生命周期的管理器（记得先运行.start()）
            array_specs: 单个时间步的数组名称、形状和类型
            get_max_k: 一次最多可查询的项目数量
            get_time_budget: 从共享内存复制数据到本地内存的最大时间（秒）
            put_desired_frequency: .put()可以被调用的最大频率（Hz）
            safety_margin: 安全边际倍数
        """
        # 创建原子计数器
        counter = SharedAtomicCounter(shm_manager)

        # 计算缓冲区大小
        # 在任何给定时刻，过去的get_max_k个项目永远不应该被触及（可以自由读取）
        # 假设读取这k个项目最多需要get_time_budget秒，
        # 我们需要足够的空槽来确保put_desired_frequency Hz的put操作可以持续
        buffer_size = int(np.ceil(
            put_desired_frequency * get_time_budget 
            * safety_margin)) + get_max_k

        # 分配共享内存
        shared_arrays = dict()
        for spec in array_specs:
            key = spec.name
            assert key not in shared_arrays
            array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(buffer_size,) + tuple(spec.shape),
                dtype=spec.dtype)
            shared_arrays[key] = array
        
        # 分配时间戳数组
        timestamp_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, 
            shape=(buffer_size,),
            dtype=np.float64)
        timestamp_array.get()[:] = -np.inf
        
        self.buffer_size = buffer_size
        self.array_specs = array_specs
        self.counter = counter
        self.shared_arrays = shared_arrays
        self.timestamp_array = timestamp_array
        self.get_time_budget = get_time_budget
        self.get_max_k = get_max_k
        self.put_desired_frequency = put_desired_frequency

    @property
    def count(self):
        """获取当前计数"""
        return self.counter.load()
    
    @classmethod
    def create_from_examples(cls, 
                             shm_manager: SharedMemoryManager,
                             examples: Dict[str, Union[np.ndarray, numbers.Number]], 
                             get_max_k: int = 32,
                             get_time_budget: float = 0.01,
                             put_desired_frequency: float = 60
                             ):
        """
        从示例数据创建环形缓冲区
        
        Args:
            shm_manager: 共享内存管理器
            examples: 示例数据字典
            get_max_k: 最大查询数量
            get_time_budget: 时间预算
            put_desired_frequency: 期望的放入频率
            
        Returns:
            创建的环形缓冲区实例
        """
        specs = list()
        for key, value in examples.items():
            shape = None
            dtype = None
            if isinstance(value, np.ndarray):
                shape = value.shape
                dtype = value.dtype
                assert dtype != np.dtype('O')
            elif isinstance(value, numbers.Number):
                shape = tuple()
                dtype = np.dtype(type(value))
            else:
                raise TypeError(f'Unsupported type {type(value)}')

            spec = ArraySpec(
                name=key,
                shape=shape,
                dtype=dtype
            )
            specs.append(spec)

        obj = cls(
            shm_manager=shm_manager,
            array_specs=specs,
            get_max_k=get_max_k,
            get_time_budget=get_time_budget,
            put_desired_frequency=put_desired_frequency
        )
        return obj

    def clear(self):
        """清空缓冲区"""
        self.counter.store(0)
    
    def put(self, data: Dict[str, Union[np.ndarray, numbers.Number]], wait: bool = True):
        """
        向缓冲区放入数据
        
        Args:
            data: 要放入的数据字典
            wait: 如果放入过快，是否等待
            
        Raises:
            TimeoutError: 当wait=False且放入过快时抛出异常
        """
        count = self.counter.load()
        next_idx = count % self.buffer_size
        
        # 确保环形缓冲区中的下self.get_max_k个元素在写入后至少有
        # self.get_time_budget秒不被触及，这样get_last_k可以安全地
        # 从任何count位置读取k个元素
        timestamp_lookahead_idx = (next_idx + self.get_max_k - 1) % self.buffer_size
        old_timestamp = self.timestamp_array.get()[timestamp_lookahead_idx]
        t = time.monotonic()
        
        if (t - old_timestamp) < self.get_time_budget:
            deltat = t - old_timestamp
            if wait:
                # 等待剩余时间以确保安全
                time.sleep(self.get_time_budget - deltat)
            else:
                # 抛出错误
                past_iters = self.buffer_size - self.get_max_k
                hz = past_iters / deltat
                raise TimeoutError(
                    'Put executed too fast {}items/{:.4f}s ~= {}Hz'.format(
                        past_iters, deltat, hz))

        # 写入共享内存
        for key, value in data.items():
            arr: np.ndarray
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                arr[next_idx] = value
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)
        
        # 更新时间戳
        self.timestamp_array.get()[next_idx] = time.monotonic()
        self.counter.add(1)

    def _allocate_empty(self, k=None):
        """分配空的输出缓冲区"""
        result = dict()
        for spec in self.array_specs:
            shape = spec.shape
            if k is not None:
                shape = (k,) + shape
            result[spec.name] = np.empty(
                shape=shape, dtype=spec.dtype)
        return result

    def get(self, out=None) -> Dict[str, np.ndarray]:
        """
        获取最新数据
        
        Args:
            out: 可选的输出缓冲区
            
        Returns:
            最新的数据字典
            
        Raises:
            TimeoutError: 当获取时间超过预算时抛出异常
        """
        if out is None:
            out = self._allocate_empty()
        start_time = time.monotonic()
        count = self.counter.load()
        curr_idx = (count - 1) % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            np.copyto(out[key], arr[curr_idx])
        end_time = time.monotonic()
        dt = end_time - start_time
        if dt > self.get_time_budget:
            raise TimeoutError(f'Get time out {dt} vs {self.get_time_budget}')
        return out
    
    def get_last_k(self, k: int, out=None) -> Dict[str, np.ndarray]:
        """
        获取最新的k个数据
        
        Args:
            k: 要获取的数据数量
            out: 可选的输出缓冲区
            
        Returns:
            最新k个数据的字典
            
        Raises:
            TimeoutError: 当获取时间超过预算时抛出异常
        """
        assert k <= self.get_max_k
        if out is None:
            out = self._allocate_empty(k)
        start_time = time.monotonic()
        count = self.counter.load()
        assert k <= count
        curr_idx = (count - 1) % self.buffer_size
        
        for key, value in self.shared_arrays.items():
            arr = value.get()
            target = out[key]

            end = curr_idx + 1
            start = max(0, end - k)
            target_end = k
            target_start = target_end - (end - start)
            target[target_start: target_end] = arr[start:end]

            remainder = k - (end - start)
            if remainder > 0:
                # 环绕处理
                end = self.buffer_size
                start = end - remainder
                target_start = 0
                target_end = end - start
                target[target_start: target_end] = arr[start:end]
                
        end_time = time.monotonic()
        dt = end_time - start_time
        if dt > self.get_time_budget:
            raise TimeoutError(f'Get time out {dt} vs {self.get_time_budget}')
        return out

    def get_all(self) -> Dict[str, np.ndarray]:
        """
        获取所有可用数据
        
        Returns:
            所有可用数据的字典
        """
        k = min(self.count, self.get_max_k)
        return self.get_last_k(k=k)
