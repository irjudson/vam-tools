"""
GPU detection and capability utilities.

Detects available GPU acceleration backends and provides information
for optimal configuration.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about available GPU acceleration."""

    available: bool = False
    device_name: Optional[str] = None
    memory_gb: float = 0.0
    cuda_version: Optional[str] = None
    compute_capability: Optional[tuple] = None
    backends: List[str] = field(default_factory=list)

    @property
    def recommended_batch_size(self) -> int:
        """
        Recommend batch size based on GPU memory.

        Returns:
            Recommended batch size for image processing
        """
        if not self.available:
            return 1

        # Rule of thumb: ~100MB per image in batch
        # Reserve 2GB for other operations
        available_memory = max(0, self.memory_gb - 2.0)
        batch_size = int(available_memory * 10)  # 100MB per image

        # Clamp to reasonable range
        return max(16, min(batch_size, 256))

    def supports_backend(self, backend: str) -> bool:
        """
        Check if a specific backend is available.

        Args:
            backend: Backend name ("cuda", "opencl", etc.)

        Returns:
            True if backend is available
        """
        return backend.lower() in [b.lower() for b in self.backends]


def detect_gpu() -> GPUInfo:
    """
    Detect available GPU and its capabilities.

    Tries multiple backends in order of preference:
    1. CUDA (NVIDIA)
    2. ROCm (AMD)
    3. OpenCL (cross-platform)

    Returns:
        GPUInfo object with detection results
    """
    gpu_info = GPUInfo()

    # Try CUDA (NVIDIA)
    if _detect_cuda(gpu_info):
        logger.info(
            f"CUDA GPU detected: {gpu_info.device_name} "
            f"({gpu_info.memory_gb:.1f}GB, CUDA {gpu_info.cuda_version})"
        )
        return gpu_info

    # Try ROCm (AMD)
    if _detect_rocm(gpu_info):
        logger.info(
            f"ROCm GPU detected: {gpu_info.device_name} ({gpu_info.memory_gb:.1f}GB)"
        )
        return gpu_info

    # Try OpenCL (cross-platform)
    if _detect_opencl(gpu_info):
        logger.info(
            f"OpenCL GPU detected: {gpu_info.device_name} ({gpu_info.memory_gb:.1f}GB)"
        )
        return gpu_info

    logger.info("No GPU detected, using CPU")
    return gpu_info


def _detect_cuda(gpu_info: GPUInfo) -> bool:
    """
    Detect CUDA-capable NVIDIA GPU.

    Args:
        gpu_info: GPUInfo object to populate

    Returns:
        True if CUDA GPU detected
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)

            gpu_info.available = True
            gpu_info.device_name = torch.cuda.get_device_name(0)
            gpu_info.memory_gb = device_props.total_memory / (1024**3)
            gpu_info.cuda_version = torch.version.cuda
            gpu_info.compute_capability = (
                device_props.major,
                device_props.minor,
            )
            gpu_info.backends.append("cuda")

            return True
    except ImportError:
        logger.debug("PyTorch not available, skipping CUDA detection")
    except Exception as e:
        logger.debug(f"Error detecting CUDA: {e}")

    return False


def _detect_rocm(gpu_info: GPUInfo) -> bool:
    """
    Detect ROCm-capable AMD GPU.

    Args:
        gpu_info: GPUInfo object to populate

    Returns:
        True if ROCm GPU detected
    """
    try:
        import torch

        if hasattr(torch, "hip") and torch.hip.is_available():
            gpu_info.available = True
            gpu_info.device_name = torch.hip.get_device_name(0)
            # ROCm/HIP doesn't expose memory the same way
            gpu_info.memory_gb = 0.0
            gpu_info.backends.append("rocm")
            return True
    except ImportError:
        logger.debug("PyTorch ROCm not available")
    except Exception as e:
        logger.debug(f"Error detecting ROCm: {e}")

    return False


def _detect_opencl(gpu_info: GPUInfo) -> bool:
    """
    Detect OpenCL-capable GPU.

    Args:
        gpu_info: GPUInfo object to populate

    Returns:
        True if OpenCL GPU detected
    """
    try:
        import pyopencl as cl

        platforms = cl.get_platforms()
        if not platforms:
            return False

        # Get first GPU device
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                device = devices[0]
                gpu_info.available = True
                gpu_info.device_name = device.name.strip()
                gpu_info.memory_gb = device.global_mem_size / (1024**3)
                gpu_info.backends.append("opencl")
                return True

    except ImportError:
        logger.debug("PyOpenCL not available")
    except Exception as e:
        logger.debug(f"Error detecting OpenCL: {e}")

    return False


def get_optimal_config(gpu_info: GPUInfo) -> Dict[str, Any]:
    """
    Get optimal configuration based on GPU capabilities.

    Args:
        gpu_info: GPU information

    Returns:
        Dictionary with recommended configuration
    """
    config = {
        "use_gpu": gpu_info.available,
        "batch_size": gpu_info.recommended_batch_size,
        "num_workers": 4,  # For CPU fallback
    }

    if gpu_info.available:
        # Adjust based on GPU memory
        if gpu_info.memory_gb >= 16:
            config["batch_size"] = 64
            config["num_workers"] = 8
        elif gpu_info.memory_gb >= 8:
            config["batch_size"] = 32
            config["num_workers"] = 4
        elif gpu_info.memory_gb >= 4:
            config["batch_size"] = 16
            config["num_workers"] = 2
        else:
            config["batch_size"] = 8
            config["num_workers"] = 1

    return config


if __name__ == "__main__":
    """Test GPU detection."""
    import json

    logging.basicConfig(level=logging.INFO)

    print("Detecting GPU capabilities...\n")
    gpu_info = detect_gpu()

    print("GPU Information:")
    print(f"  Available: {gpu_info.available}")
    if gpu_info.available:
        print(f"  Device: {gpu_info.device_name}")
        print(f"  Memory: {gpu_info.memory_gb:.2f} GB")
        print(f"  Backends: {', '.join(gpu_info.backends)}")
        if gpu_info.cuda_version:
            print(f"  CUDA Version: {gpu_info.cuda_version}")
        if gpu_info.compute_capability:
            print(f"  Compute Capability: {gpu_info.compute_capability}")

    print("\nRecommended Configuration:")
    config = get_optimal_config(gpu_info)
    print(json.dumps(config, indent=2))
