"""
共享NumPy数组实现

提供基于共享内存的NumPy数组接口。
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Any, TYPE_CHECKING, Generic, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
from .nested_dict_util import (nested_dict_check, nested_dict_map)


SharedMemoryLike = Union[str, SharedMemory]  # shared memory or name of shared memory
SharedT = TypeVar("SharedT", bound=np.generic)


class SharedNDArray(Generic[SharedT]):
    """
    共享NumPy数组类
    
    用于跟踪和检索共享数组中的数据。支持通过共享内存在多进程间共享NumPy数组。
    
    Attributes:
        shm: 包含数组数据的SharedMemory对象
        shape: NumPy数组的形状
        dtype: NumPy数组的数据类型
        lock: （可选）用于管理访问的多进程锁
    
    Examples:
        >>> from multiprocessing.managers import SharedMemoryManager
        >>> import numpy as np
        >>> 
        >>> mem_mgr = SharedMemoryManager()
        >>> mem_mgr.start()
        >>> 
        >>> # 从现有数组创建
        >>> x = np.array([1, 2, 3])
        >>> arr = SharedNDArray.create_from_array(mem_mgr, x)
        >>> print(arr[:])
        [1 2 3]
        >>> 
        >>> # 从形状创建
        >>> arr = SharedNDArray.create_from_shape(mem_mgr, (3,), np.int32)
        >>> arr[:] = [4, 5, 6]
        >>> print(arr[:])
        [4 5 6]
        >>> 
        >>> mem_mgr.shutdown()
    """

    shm: SharedMemory
    dtype: np.dtype
    lock: Optional[multiprocessing.synchronize.Lock]

    def __init__(
        self, shm: SharedMemoryLike, shape: Tuple[int, ...], dtype: npt.DTypeLike):
        """
        从现有共享内存、对象形状和数据类型初始化SharedNDArray对象
        
        Parameters:
            shm: SharedMemory对象或连接到现有共享内存的名称
            shape: 共享内存中NumPy数组的形状
            dtype: 共享内存中NumPy数组的数据类型
            
        Raises:
            ValueError: 共享内存大小与形状和dtype不匹配
        """
        if isinstance(shm, str):
            shm = SharedMemory(name=shm, create=False)
        dtype = np.dtype(dtype)  # 尝试转换为dtype
        assert shm.size >= (dtype.itemsize * np.prod(shape))
        self.shm = shm
        self.dtype = dtype
        self._shape: Tuple[int, ...] = shape

    def __repr__(self):
        """返回数组的字符串表示"""
        cls_name = self.__class__.__name__
        nspaces = len(cls_name) + 1
        array_repr = str(self.get())
        array_repr = array_repr.replace("\n", "\n" + " " * nspaces)
        return f"{cls_name}({array_repr}, dtype={self.dtype})"

    @classmethod
    def create_from_array(
        cls, mem_mgr: SharedMemoryManager, arr: npt.NDArray[SharedT]
    ) -> SharedNDArray[SharedT]:
        """
        从SharedMemoryManager和现有numpy数组创建SharedNDArray
        
        Parameters:
            mem_mgr: 运行中的SharedMemoryManager实例
            arr: 要复制到SharedNDArray中的NumPy数组
        """
        shared_arr = cls.create_from_shape(mem_mgr, arr.shape, arr.dtype)
        shared_arr.get()[:] = arr[:]
        return shared_arr

    @classmethod
    def create_from_shape(
        cls, mem_mgr: SharedMemoryManager, shape: Tuple, dtype: npt.DTypeLike) -> SharedNDArray:
        """
        直接从SharedMemoryManager创建SharedNDArray
        
        Parameters:
            mem_mgr: 已启动的SharedMemoryManager实例
            shape: 数组的形状
            dtype: 数组的数据类型
        """
        dtype = np.dtype(dtype)  # 如果可能，转换为dtype
        shm = mem_mgr.SharedMemory(np.prod(shape) * dtype.itemsize)
        return cls(shm=shm, shape=shape, dtype=dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        """获取数组形状"""
        return self._shape

    def get(self) -> npt.NDArray[SharedT]:
        """获取具有共享内存访问权限的numpy数组"""
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def __del__(self):
        """析构函数，关闭共享内存"""
        self.shm.close()
