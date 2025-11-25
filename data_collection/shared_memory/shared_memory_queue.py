"""
共享内存队列实现

提供无锁FIFO共享内存数据结构，用于存储numpy数组序列。
"""

from typing import Dict, List, Union
import numbers
from queue import (Empty, Full)
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from .shared_memory_util import ArraySpec, SharedAtomicCounter
from .shared_ndarray import SharedNDArray


class SharedMemoryQueue:
    """
    无锁FIFO共享内存数据结构
    存储numpy数组字典的序列
    
    这个类实现了一个高效的进程间通信队列，特别适合实时系统中的数据传输。
    
    Examples:
        >>> from multiprocessing.managers import SharedMemoryManager
        >>> import numpy as np
        >>> 
        >>> with SharedMemoryManager() as shm_manager:
        ...     # 从示例数据创建队列
        ...     examples = {
        ...         'position': np.array([1.0, 2.0, 3.0]),
        ...         'velocity': np.array([0.1, 0.2, 0.3])
        ...     }
        ...     queue = SharedMemoryQueue.create_from_examples(
        ...         shm_manager, examples, buffer_size=10)
        ...     
        ...     # 放入数据
        ...     data = {'position': np.array([4.0, 5.0, 6.0]), 
        ...             'velocity': np.array([0.4, 0.5, 0.6])}
        ...     queue.put(data)
        ...     
        ...     # 获取数据
        ...     result = queue.get()
        ...     print(result['position'])  # [4. 5. 6.]
    """

    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 array_specs: List[ArraySpec],
                 buffer_size: int
                 ):
        """
        初始化共享内存队列
        
        Args:
            shm_manager: 共享内存管理器
            array_specs: 数组规格列表
            buffer_size: 缓冲区大小
        """
        # 创建原子计数器
        write_counter = SharedAtomicCounter(shm_manager)
        read_counter = SharedAtomicCounter(shm_manager)
        
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
        
        self.buffer_size = buffer_size
        self.array_specs = array_specs
        self.write_counter = write_counter
        self.read_counter = read_counter
        self.shared_arrays = shared_arrays
    
    @classmethod
    def create_from_examples(cls, 
                             shm_manager: SharedMemoryManager,
                             examples: Dict[str, Union[np.ndarray, numbers.Number]], 
                             buffer_size: int
                             ):
        """
        从示例数据创建队列
        
        Args:
            shm_manager: 共享内存管理器
            examples: 示例数据字典
            buffer_size: 缓冲区大小
            
        Returns:
            创建的队列实例
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
            buffer_size=buffer_size
        )
        return obj
    
    def qsize(self):
        """获取队列中的数据数量"""
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        return n_data
    
    def empty(self):
        """检查队列是否为空"""
        n_data = self.qsize()
        return n_data <= 0
    
    def clear(self):
        """清空队列"""
        self.read_counter.store(self.write_counter.load())
    
    def put(self, data: Dict[str, Union[np.ndarray, numbers.Number]]):
        """
        向队列中放入数据
        
        Args:
            data: 要放入的数据字典
            
        Raises:
            Full: 队列已满时抛出异常
        """
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        if n_data >= self.buffer_size:
            raise Full()
        
        next_idx = write_count % self.buffer_size

        # 写入共享内存
        for key, value in data.items():
            arr: np.ndarray
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                arr[next_idx] = value
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)

        # 更新索引
        self.write_counter.add(1)
    
    def get(self, out=None) -> Dict[str, np.ndarray]:
        """
        从队列中获取数据
        
        Args:
            out: 可选的输出缓冲区
            
        Returns:
            获取的数据字典
            
        Raises:
            Empty: 队列为空时抛出异常
        """
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        if out is None:
            out = self._allocate_empty()

        next_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            np.copyto(out[key], arr[next_idx])
        
        # 更新索引
        self.read_counter.add(1)
        return out

    def get_k(self, k, out=None) -> Dict[str, np.ndarray]:
        """
        从队列中获取k个数据
        
        Args:
            k: 要获取的数据数量
            out: 可选的输出缓冲区
            
        Returns:
            获取的数据字典
        """
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()
        assert k <= n_data

        out = self._get_k_impl(k, read_count, out=out)
        self.read_counter.add(k)
        return out

    def get_all(self, out=None) -> Dict[str, np.ndarray]:
        """
        获取队列中的所有数据
        
        Args:
            out: 可选的输出缓冲区
            
        Returns:
            获取的所有数据字典
        """
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        out = self._get_k_impl(n_data, read_count, out=out)
        self.read_counter.add(n_data)
        return out
    
    def peek_all(self, out=None) -> Dict[str, np.ndarray]:
       """
       查看队列中的所有数据（非消费性读取）


       Args:
           out: 可选的输出缓冲区


       Returns:
           获取的所有数据字典，不移动读指针
       """
       write_count = self.write_counter.load()
       read_count = self.read_counter.load()
       n_data = write_count - read_count
       if n_data <= 0:
           raise Empty()


       out = self._get_k_impl(n_data, read_count, out=out)
       # 注意：这里不调用 self.read_counter.add(n_data)
       return out
    
    def _get_k_impl(self, k, read_count, out=None) -> Dict[str, np.ndarray]:
        """获取k个数据的内部实现"""
        if out is None:
            out = self._allocate_empty(k)

        curr_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            target = out[key]

            start = curr_idx
            end = min(start + k, self.buffer_size)
            target_start = 0
            target_end = (end - start)
            target[target_start: target_end] = arr[start:end]

            remainder = k - (end - start)
            if remainder > 0:
                # 环绕处理
                start = 0
                end = start + remainder
                target_start = target_end
                target_end = k
                target[target_start: target_end] = arr[start:end]

        return out
    
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
