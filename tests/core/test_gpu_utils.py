"""
Tests for GPU detection and utility functions.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from vam_tools.core.gpu_utils import (
    GPUInfo,
    detect_gpu,
    get_optimal_config,
)


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_gpu_info_default_values(self) -> None:
        """Test default GPUInfo initialization."""
        info = GPUInfo()

        assert info.available is False
        assert info.device_name is None
        assert info.memory_gb == 0.0
        assert info.cuda_version is None
        assert info.compute_capability is None
        assert info.backends == []

    def test_gpu_info_with_values(self) -> None:
        """Test GPUInfo with explicit values."""
        info = GPUInfo(
            available=True,
            device_name="NVIDIA RTX 3080",
            memory_gb=10.0,
            cuda_version="11.8",
            compute_capability=(8, 6),
            backends=["cuda"],
        )

        assert info.available is True
        assert info.device_name == "NVIDIA RTX 3080"
        assert info.memory_gb == 10.0
        assert info.cuda_version == "11.8"
        assert info.compute_capability == (8, 6)
        assert info.backends == ["cuda"]

    def test_recommended_batch_size_no_gpu(self) -> None:
        """Test batch size recommendation without GPU."""
        info = GPUInfo(available=False)

        assert info.recommended_batch_size == 1

    def test_recommended_batch_size_small_gpu(self) -> None:
        """Test batch size for small GPU (4GB)."""
        info = GPUInfo(available=True, memory_gb=4.0)

        # (4.0 - 2.0) * 10 = 20, clamped to [16, 256]
        assert info.recommended_batch_size == 20

    def test_recommended_batch_size_medium_gpu(self) -> None:
        """Test batch size for medium GPU (8GB)."""
        info = GPUInfo(available=True, memory_gb=8.0)

        # (8.0 - 2.0) * 10 = 60
        assert info.recommended_batch_size == 60

    def test_recommended_batch_size_large_gpu(self) -> None:
        """Test batch size for large GPU (24GB)."""
        info = GPUInfo(available=True, memory_gb=24.0)

        # (24.0 - 2.0) * 10 = 220
        assert info.recommended_batch_size == 220

    def test_recommended_batch_size_huge_gpu(self) -> None:
        """Test batch size clamped at maximum (256)."""
        info = GPUInfo(available=True, memory_gb=50.0)

        # (50.0 - 2.0) * 10 = 480, clamped to 256
        assert info.recommended_batch_size == 256

    def test_recommended_batch_size_tiny_gpu(self) -> None:
        """Test batch size clamped at minimum (16)."""
        info = GPUInfo(available=True, memory_gb=2.5)

        # (2.5 - 2.0) * 10 = 5, clamped to 16
        assert info.recommended_batch_size == 16

    def test_supports_backend_true(self) -> None:
        """Test backend support check when backend is available."""
        info = GPUInfo(backends=["cuda", "opencl"])

        assert info.supports_backend("cuda") is True
        assert info.supports_backend("CUDA") is True  # Case insensitive
        assert info.supports_backend("opencl") is True

    def test_supports_backend_false(self) -> None:
        """Test backend support check when backend is not available."""
        info = GPUInfo(backends=["cuda"])

        assert info.supports_backend("rocm") is False
        assert info.supports_backend("opencl") is False


class TestDetectGPU:
    """Tests for GPU detection functions."""

    def test_detect_gpu_no_backends(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test GPU detection when no backends are available."""
        # Mock all detection functions to return False
        with patch("vam_tools.core.gpu_utils._detect_cuda", return_value=False):
            with patch("vam_tools.core.gpu_utils._detect_rocm", return_value=False):
                with patch(
                    "vam_tools.core.gpu_utils._detect_opencl", return_value=False
                ):
                    gpu_info = detect_gpu()

        assert gpu_info.available is False
        assert len(gpu_info.backends) == 0

    def test_detect_gpu_cuda_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test GPU detection when CUDA is available."""

        def mock_detect_cuda(gpu_info: GPUInfo) -> bool:
            gpu_info.available = True
            gpu_info.device_name = "NVIDIA RTX 3080"
            gpu_info.memory_gb = 10.0
            gpu_info.cuda_version = "11.8"
            gpu_info.backends.append("cuda")
            return True

        with patch(
            "vam_tools.core.gpu_utils._detect_cuda", side_effect=mock_detect_cuda
        ):
            gpu_info = detect_gpu()

        assert gpu_info.available is True
        assert gpu_info.device_name == "NVIDIA RTX 3080"
        assert "cuda" in gpu_info.backends

    def test_detect_gpu_rocm_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test GPU detection falls back to ROCm when CUDA unavailable."""

        def mock_detect_rocm(gpu_info: GPUInfo) -> bool:
            gpu_info.available = True
            gpu_info.device_name = "AMD Radeon RX 6800"
            gpu_info.memory_gb = 16.0
            gpu_info.backends.append("rocm")
            return True

        with patch("vam_tools.core.gpu_utils._detect_cuda", return_value=False):
            with patch(
                "vam_tools.core.gpu_utils._detect_rocm", side_effect=mock_detect_rocm
            ):
                gpu_info = detect_gpu()

        assert gpu_info.available is True
        assert gpu_info.device_name == "AMD Radeon RX 6800"
        assert "rocm" in gpu_info.backends

    def test_detect_gpu_opencl_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test GPU detection falls back to OpenCL."""

        def mock_detect_opencl(gpu_info: GPUInfo) -> bool:
            gpu_info.available = True
            gpu_info.device_name = "Intel UHD Graphics"
            gpu_info.memory_gb = 4.0
            gpu_info.backends.append("opencl")
            return True

        with patch("vam_tools.core.gpu_utils._detect_cuda", return_value=False):
            with patch("vam_tools.core.gpu_utils._detect_rocm", return_value=False):
                with patch(
                    "vam_tools.core.gpu_utils._detect_opencl",
                    side_effect=mock_detect_opencl,
                ):
                    gpu_info = detect_gpu()

        assert gpu_info.available is True
        assert "opencl" in gpu_info.backends


class TestDetectCUDA:
    """Tests for CUDA detection."""

    def test_detect_cuda_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test CUDA detection when available."""
        from vam_tools.core.gpu_utils import _detect_cuda

        # Mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3080"
        mock_torch.version.cuda = "11.8"

        # Mock device properties
        mock_props = MagicMock()
        mock_props.total_memory = 10 * (1024**3)  # 10GB
        mock_props.major = 8
        mock_props.minor = 6
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "torch", mock_torch)

            gpu_info = GPUInfo()
            result = _detect_cuda(gpu_info)

        assert result is True
        assert gpu_info.available is True
        assert gpu_info.device_name == "NVIDIA GeForce RTX 3080"
        assert gpu_info.memory_gb == 10.0
        assert gpu_info.cuda_version == "11.8"
        assert gpu_info.compute_capability == (8, 6)
        assert "cuda" in gpu_info.backends

    def test_detect_cuda_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test CUDA detection when CUDA not available."""
        from vam_tools.core.gpu_utils import _detect_cuda

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "torch", mock_torch)

            gpu_info = GPUInfo()
            result = _detect_cuda(gpu_info)

        assert result is False
        assert gpu_info.available is False

    def test_detect_cuda_torch_not_installed(self) -> None:
        """Test CUDA detection when PyTorch is not installed."""
        from vam_tools.core.gpu_utils import _detect_cuda

        # Skip if torch is already loaded
        try:
            pass

            pytest.skip("PyTorch is already loaded, cannot test unavailable case")
        except ImportError:
            pass

        gpu_info = GPUInfo()
        result = _detect_cuda(gpu_info)

        # Should return False gracefully
        assert result is False
        assert gpu_info.available is False

    def test_detect_cuda_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test CUDA detection handles exceptions gracefully."""
        from vam_tools.core.gpu_utils import _detect_cuda

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA error")

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "torch", mock_torch)

            gpu_info = GPUInfo()
            result = _detect_cuda(gpu_info)

        assert result is False
        assert gpu_info.available is False


class TestDetectROCm:
    """Tests for ROCm detection."""

    def test_detect_rocm_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ROCm detection when available."""
        from vam_tools.core.gpu_utils import _detect_rocm

        mock_torch = MagicMock()
        mock_torch.hip.is_available.return_value = True
        mock_torch.hip.get_device_name.return_value = "AMD Radeon RX 6800 XT"

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "torch", mock_torch)

            gpu_info = GPUInfo()
            result = _detect_rocm(gpu_info)

        assert result is True
        assert gpu_info.available is True
        assert gpu_info.device_name == "AMD Radeon RX 6800 XT"
        assert "rocm" in gpu_info.backends

    def test_detect_rocm_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ROCm detection when not available."""
        from vam_tools.core.gpu_utils import _detect_rocm

        mock_torch = MagicMock()
        delattr(mock_torch, "hip")  # torch.hip doesn't exist

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "torch", mock_torch)

            gpu_info = GPUInfo()
            result = _detect_rocm(gpu_info)

        assert result is False
        assert gpu_info.available is False


class TestDetectOpenCL:
    """Tests for OpenCL detection."""

    def test_detect_opencl_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test OpenCL detection when available."""
        from vam_tools.core.gpu_utils import _detect_opencl

        # Mock pyopencl
        mock_cl = MagicMock()
        mock_device = MagicMock()
        mock_device.name = "  Intel UHD Graphics  "
        mock_device.global_mem_size = 4 * (1024**3)  # 4GB

        mock_platform = MagicMock()
        mock_platform.get_devices.return_value = [mock_device]

        mock_cl.get_platforms.return_value = [mock_platform]
        mock_cl.device_type.GPU = 2  # Mock constant

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "pyopencl", mock_cl)

            gpu_info = GPUInfo()
            result = _detect_opencl(gpu_info)

        assert result is True
        assert gpu_info.available is True
        assert gpu_info.device_name == "Intel UHD Graphics"
        assert gpu_info.memory_gb == 4.0
        assert "opencl" in gpu_info.backends

    def test_detect_opencl_no_platforms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test OpenCL detection with no platforms."""
        from vam_tools.core.gpu_utils import _detect_opencl

        mock_cl = MagicMock()
        mock_cl.get_platforms.return_value = []

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "pyopencl", mock_cl)

            gpu_info = GPUInfo()
            result = _detect_opencl(gpu_info)

        assert result is False
        assert gpu_info.available is False


class TestGetOptimalConfig:
    """Tests for optimal configuration generation."""

    def test_optimal_config_no_gpu(self) -> None:
        """Test config for CPU-only."""
        gpu_info = GPUInfo(available=False)

        config = get_optimal_config(gpu_info)

        assert config["use_gpu"] is False
        assert config["batch_size"] == 1
        assert config["num_workers"] == 4

    def test_optimal_config_small_gpu(self) -> None:
        """Test config for small GPU (3GB)."""
        gpu_info = GPUInfo(available=True, memory_gb=3.0)

        config = get_optimal_config(gpu_info)

        assert config["use_gpu"] is True
        assert config["batch_size"] == 8
        assert config["num_workers"] == 1

    def test_optimal_config_medium_gpu(self) -> None:
        """Test config for medium GPU (6GB)."""
        gpu_info = GPUInfo(available=True, memory_gb=6.0)

        config = get_optimal_config(gpu_info)

        assert config["use_gpu"] is True
        assert config["batch_size"] == 16
        assert config["num_workers"] == 2

    def test_optimal_config_large_gpu(self) -> None:
        """Test config for large GPU (10GB)."""
        gpu_info = GPUInfo(available=True, memory_gb=10.0)

        config = get_optimal_config(gpu_info)

        assert config["use_gpu"] is True
        assert config["batch_size"] == 32
        assert config["num_workers"] == 4

    def test_optimal_config_huge_gpu(self) -> None:
        """Test config for huge GPU (24GB)."""
        gpu_info = GPUInfo(available=True, memory_gb=24.0)

        config = get_optimal_config(gpu_info)

        assert config["use_gpu"] is True
        assert config["batch_size"] == 64
        assert config["num_workers"] == 8
