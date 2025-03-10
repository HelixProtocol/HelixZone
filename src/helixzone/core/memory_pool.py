"""Memory pool management module."""
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from numpy.typing import NDArray
import pyopencl as cl
from dataclasses import dataclass
from enum import Enum, auto
import threading
import weakref
from collections import defaultdict

@dataclass
class MemoryBlock:
    """Memory block information."""
    buffer: cl.Buffer
    size: int
    in_use: bool
    last_used: float
    access_count: int

class MemoryType(Enum):
    """Memory types for allocation."""
    HOST = auto()
    DEVICE = auto()
    UNIFIED = auto()

class MemoryPool:
    """Memory pool for efficient buffer allocation."""
    def __init__(
        self,
        device: cl.Device,
        initial_size: int = 1024 * 1024,  # 1MB
        max_size: int = 1024 * 1024 * 1024,  # 1GB
        growth_factor: float = 2.0,
        cleanup_threshold: float = 0.75,  # 75% utilization triggers cleanup
        cache_timeout: float = 60.0  # 60 seconds
    ):
        self.device = device
        self.ctx = cl.Context([device])
        self.initial_size = initial_size
        self.max_size = max_size
        self.growth_factor = growth_factor
        self.cleanup_threshold = cleanup_threshold
        self.cache_timeout = cache_timeout
        
        # Memory pools for different types
        self._pools: Dict[MemoryType, Dict[int, List[MemoryBlock]]] = {
            mem_type: defaultdict(list)
            for mem_type in MemoryType
        }
        
        # Track total allocated memory
        self._total_allocated = 0
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Track active buffers
        self._active_buffers: Set[int] = set()
        
        # Initialize pools
        self._initialize_pools()
        
    def _initialize_pools(self) -> None:
        """Initialize memory pools."""
        with self._lock:
            # Pre-allocate some common sizes
            sizes = [
                1024,      # 1KB
                4096,      # 4KB
                16384,     # 16KB
                65536,     # 64KB
                262144,    # 256KB
                1048576    # 1MB
            ]
            
            for mem_type in MemoryType:
                for size in sizes:
                    self._allocate_block(size, mem_type)
                    
    def _allocate_block(self, size: int, mem_type: MemoryType) -> MemoryBlock:
        """Allocate a new memory block."""
        import time
        
        flags = {
            MemoryType.HOST: cl.mem_flags.ALLOC_HOST_PTR,
            MemoryType.DEVICE: cl.mem_flags.READ_WRITE,
            MemoryType.UNIFIED: cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR
        }[mem_type]
        
        buffer = cl.Buffer(self.ctx, flags, size)
        block = MemoryBlock(
            buffer=buffer,
            size=size,
            in_use=False,
            last_used=time.time(),
            access_count=0
        )
        
        self._total_allocated += size
        return block
        
    def _find_best_fit(self, size: int, mem_type: MemoryType) -> Optional[MemoryBlock]:
        """Find the best fitting block for the requested size."""
        pool = self._pools[mem_type]
        
        # Try exact size first
        if size in pool:
            for block in pool[size]:
                if not block.in_use:
                    return block
                    
        # Try larger sizes
        for block_size in sorted(pool.keys()):
            if block_size >= size:
                for block in pool[block_size]:
                    if not block.in_use:
                        return block
                        
        return None
        
    def _cleanup_unused(self) -> None:
        """Clean up unused memory blocks."""
        import time
        current_time = time.time()
        
        with self._lock:
            for mem_type in MemoryType:
                pool = self._pools[mem_type]
                for size, blocks in list(pool.items()):
                    # Keep blocks that are in use or recently used
                    new_blocks = []
                    for block in blocks:
                        if (block.in_use or
                            current_time - block.last_used < self.cache_timeout or
                            block.access_count > 10):  # Keep frequently used blocks
                            new_blocks.append(block)
                        else:
                            # Free the memory
                            block.buffer.release()
                            self._total_allocated -= block.size
                            
                    if new_blocks:
                        pool[size] = new_blocks
                    else:
                        del pool[size]
                        
    def allocate(
        self,
        size: int,
        mem_type: MemoryType = MemoryType.DEVICE,
        hostbuf: Optional[NDArray] = None
    ) -> cl.Buffer:
        """Allocate a buffer from the pool."""
        import time
        
        with self._lock:
            # Check if cleanup is needed
            if self._total_allocated >= self.max_size * self.cleanup_threshold:
                self._cleanup_unused()
                
            # Try to find an existing block
            block = self._find_best_fit(size, mem_type)
            
            if block is None:
                # Allocate new block with some extra space for future use
                new_size = max(size, self.initial_size)
                while new_size < size:
                    new_size = int(new_size * self.growth_factor)
                    
                if self._total_allocated + new_size > self.max_size:
                    raise MemoryError("Memory pool exhausted")
                    
                block = self._allocate_block(new_size, mem_type)
                self._pools[mem_type][new_size].append(block)
                
            # Mark block as in use
            block.in_use = True
            block.last_used = time.time()
            block.access_count += 1
            
            # Copy host buffer if provided
            if hostbuf is not None:
                cl.enqueue_copy(self.ctx.devices[0].default_queue, block.buffer, hostbuf)
                
            # Track active buffer
            self._active_buffers.add(id(block.buffer))
            
            return block.buffer
            
    def free(self, buffer: cl.Buffer) -> None:
        """Return a buffer to the pool."""
        buffer_id = id(buffer)
        
        with self._lock:
            if buffer_id not in self._active_buffers:
                return
                
            # Find and mark the block as unused
            for mem_type in MemoryType:
                for blocks in self._pools[mem_type].values():
                    for block in blocks:
                        if id(block.buffer) == buffer_id:
                            block.in_use = False
                            self._active_buffers.remove(buffer_id)
                            return
                            
    def clear(self) -> None:
        """Clear all memory pools."""
        with self._lock:
            for mem_type in MemoryType:
                for blocks in self._pools[mem_type].values():
                    for block in blocks:
                        block.buffer.release()
                        
            self._pools = {
                mem_type: defaultdict(list)
                for mem_type in MemoryType
            }
            self._total_allocated = 0
            self._active_buffers.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            stats = {
                'total_allocated': self._total_allocated,
                'max_size': self.max_size,
                'utilization': self._total_allocated / self.max_size,
                'active_buffers': len(self._active_buffers),
                'pools': {}
            }
            
            for mem_type in MemoryType:
                pool_stats = {
                    'total_blocks': sum(len(blocks) for blocks in self._pools[mem_type].values()),
                    'used_blocks': sum(
                        sum(1 for block in blocks if block.in_use)
                        for blocks in self._pools[mem_type].values()
                    ),
                    'size_distribution': {
                        size: len(blocks)
                        for size, blocks in self._pools[mem_type].items()
                    }
                }
                stats['pools'][mem_type.name] = pool_stats
                
            return stats 