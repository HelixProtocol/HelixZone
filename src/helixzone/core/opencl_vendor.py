"""Vendor-specific OpenCL optimizations."""
from typing import Dict, Optional, Any, Tuple
import pyopencl as cl
from enum import Enum, auto

class VendorType(Enum):
    """OpenCL vendor types."""
    AMD = auto()
    NVIDIA = auto()
    INTEL = auto()
    OTHER = auto()

class VendorOptimizer:
    """Optimizer for vendor-specific OpenCL parameters."""
    def __init__(self, device: cl.Device):
        self.device = device
        self.vendor = self._detect_vendor()
        self.compute_units = device.max_compute_units
        self.local_mem_size = device.local_mem_size
        self.max_work_group_size = device.max_work_group_size
        self.vector_width = min(device.preferred_vector_width_float, 4)
        
    def _detect_vendor(self) -> VendorType:
        """Detect the vendor type."""
        vendor_name = self.device.vendor.lower()
        if 'amd' in vendor_name:
            return VendorType.AMD
        elif 'nvidia' in vendor_name:
            return VendorType.NVIDIA
        elif 'intel' in vendor_name:
            return VendorType.INTEL
        else:
            return VendorType.OTHER
            
    def get_work_group_size(self) -> Tuple[int, int]:
        """Get optimal work group size for the vendor."""
        if self.vendor == VendorType.AMD:
            # AMD GPUs work best with 16x16 work groups
            return (16, 16)
        elif self.vendor == VendorType.NVIDIA:
            # NVIDIA GPUs prefer 32x32 work groups
            return (32, 32)
        elif self.vendor == VendorType.INTEL:
            # Intel GPUs/CPUs work well with 8x8 work groups
            return (8, 8)
        else:
            # Default to 16x16 for unknown vendors
            return (16, 16)
            
    def get_local_memory_size(self) -> int:
        """Get optimal local memory size for the vendor."""
        if self.vendor == VendorType.AMD:
            # AMD GPUs have large local memory (64KB)
            return min(65536, self.local_mem_size)
        elif self.vendor == VendorType.NVIDIA:
            # NVIDIA GPUs have 48KB local memory
            return min(49152, self.local_mem_size)
        elif self.vendor == VendorType.INTEL:
            # Intel devices have varying local memory
            return min(32768, self.local_mem_size)
        else:
            # Default to 32KB for unknown vendors
            return min(32768, self.local_mem_size)
            
    def get_vector_width(self) -> int:
        """Get optimal vector width for the vendor."""
        if self.vendor == VendorType.AMD:
            # AMD GPUs work well with float4
            return min(4, self.vector_width)
        elif self.vendor == VendorType.NVIDIA:
            # NVIDIA GPUs prefer float2
            return min(2, self.vector_width)
        elif self.vendor == VendorType.INTEL:
            # Intel devices support up to float8
            return min(8, self.vector_width)
        else:
            # Default to float4 for unknown vendors
            return min(4, self.vector_width)
            
    def get_kernel_defines(self) -> str:
        """Get vendor-specific kernel defines."""
        defines = []
        
        # Add vendor-specific defines
        if self.vendor == VendorType.AMD:
            defines.extend([
                "#define USE_AMD_EXTENSIONS",
                "#define VECTOR_TYPE float4",
                "#define VECTOR_SIZE 4",
                "#define LOCAL_MEM_BANKS 32"
            ])
        elif self.vendor == VendorType.NVIDIA:
            defines.extend([
                "#define USE_NVIDIA_EXTENSIONS",
                "#define VECTOR_TYPE float2",
                "#define VECTOR_SIZE 2",
                "#define LOCAL_MEM_BANKS 16"
            ])
        elif self.vendor == VendorType.INTEL:
            defines.extend([
                "#define USE_INTEL_EXTENSIONS",
                "#define VECTOR_TYPE float8",
                "#define VECTOR_SIZE 8",
                "#define LOCAL_MEM_BANKS 16"
            ])
        else:
            defines.extend([
                "#define VECTOR_TYPE float4",
                "#define VECTOR_SIZE 4",
                "#define LOCAL_MEM_BANKS 16"
            ])
            
        # Add common defines
        defines.extend([
            f"#define MAX_COMPUTE_UNITS {self.compute_units}",
            f"#define LOCAL_MEM_SIZE {self.get_local_memory_size()}",
            f"#define MAX_WORK_GROUP_SIZE {self.max_work_group_size}"
        ])
        
        return "\n".join(defines)
        
    def get_kernel_options(self) -> Dict[str, Any]:
        """Get vendor-specific kernel compilation options."""
        options = {
            "-cl-mad-enable": True,  # Enable multiply-add fusion
            "-cl-no-signed-zeros": True,  # Ignore sign of zero
            "-cl-finite-math-only": True,  # No infinities or NaNs
        }
        
        if self.vendor == VendorType.AMD:
            options.update({
                "-cl-denorms-are-zero": True,  # Flush denormals to zero
                "-cl-fast-relaxed-math": True  # Fast math optimizations
            })
        elif self.vendor == VendorType.NVIDIA:
            options.update({
                "-cl-nv-verbose": True,  # Enable verbose NVIDIA compiler output
                "-cl-nv-maxrregcount=32": True  # Limit register usage
            })
        elif self.vendor == VendorType.INTEL:
            options.update({
                "-cl-intel-greater-than-4GB-buffer-required": True,  # Support large buffers
                "-cl-uniform-work-group-size": True  # Uniform work group size
            })
            
        return options
        
    def optimize_kernel_source(self, source: str) -> str:
        """Optimize kernel source for the vendor."""
        # Add vendor-specific defines
        defines = self.get_kernel_defines()
        
        # Add vendor-specific pragmas and attributes
        pragmas = []
        if self.vendor == VendorType.AMD:
            pragmas.extend([
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable",
                "#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable"
            ])
        elif self.vendor == VendorType.NVIDIA:
            pragmas.extend([
                "#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable"
            ])
        elif self.vendor == VendorType.INTEL:
            pragmas.extend([
                "#pragma OPENCL EXTENSION cl_intel_subgroups : enable",
                "#pragma OPENCL EXTENSION cl_intel_media_block_io : enable"
            ])
            
        # Combine all parts
        return "\n".join([defines, "\n".join(pragmas), source])
        
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Get optimal parameters for the vendor."""
        return {
            "work_group_size": self.get_work_group_size(),
            "local_memory_size": self.get_local_memory_size(),
            "vector_width": self.get_vector_width(),
            "kernel_options": self.get_kernel_options()
        } 