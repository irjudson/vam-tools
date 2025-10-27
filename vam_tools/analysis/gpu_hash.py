"""
GPU-accelerated perceptual hash computation.

Provides batch processing of perceptual hashes using PyTorch CUDA acceleration.
Falls back gracefully to CPU if GPU is not available.
"""
# mypy: disable-error-code="assignment,no-any-return,misc,list-item,return-value,unused-ignore,name-defined"

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from PIL import Image

from vam_tools.core.gpu_utils import detect_gpu

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class GPUHashProcessor:
    """GPU-accelerated batch hash computation."""

    def __init__(
        self,
        batch_size: Optional[int] = None,
        device: str = "cuda",
        enable_gpu: bool = True,
    ):
        """
        Initialize GPU hash processor.

        Args:
            batch_size: Number of images to process in parallel (None = auto-detect)
            device: Device to use ("cuda" or "cpu")
            enable_gpu: Whether to enable GPU acceleration
        """
        self.gpu_info = detect_gpu()
        self.use_gpu = enable_gpu and self.gpu_info.available and device == "cuda"

        if self.use_gpu:
            try:
                import torch

                self.torch = torch
                self.device = torch.device("cuda")

                # Auto-detect optimal batch size
                if batch_size is None:
                    batch_size = self.gpu_info.recommended_batch_size

                self.batch_size = batch_size
                logger.info(
                    f"GPU hash processor initialized: {self.gpu_info.device_name}, "
                    f"batch_size={self.batch_size}"
                )
            except ImportError:
                logger.warning("PyTorch not available, falling back to CPU")
                self.use_gpu = False
        else:
            logger.info("GPU not available or disabled, using CPU")

        if not self.use_gpu:
            self.batch_size = 1
            self.device = None
            self.torch = None

    def _load_and_preprocess(
        self, image_paths: List[Path], target_size: int = 9
    ) -> Optional["torch.Tensor"]:  # type: ignore[name-defined]
        """
        Load and preprocess images on GPU.

        Args:
            image_paths: List of image file paths
            target_size: Target size for resizing (9 for dhash/ahash, 8 for whash)

        Returns:
            Tensor of preprocessed images (batch, 1, target_size, target_size) or None
        """
        if not self.use_gpu:
            return None

        import torchvision.transforms as transforms

        # Define preprocessing pipeline
        preprocess = transforms.Compose(
            [
                transforms.Resize((target_size, target_size), antialias=True),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )

        batch_tensors = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                if img.mode == "RGBA":
                    # Convert RGBA to RGB
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                tensor = preprocess(img)
                batch_tensors.append(tensor)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                # Add zero tensor as placeholder
                batch_tensors.append(
                    self.torch.zeros(1, target_size, target_size, device=self.device)
                )

        if not batch_tensors:
            return None

        # Stack into batch and move to GPU
        batch = self.torch.stack(batch_tensors).to(self.device)
        return batch

    def _compute_dhash_batch(self, images: "torch.Tensor") -> List[str]:  # type: ignore[name-defined]
        """
        Compute dHash for a batch of images on GPU.

        Args:
            images: Batch of preprocessed images (batch, 1, 9, 9)

        Returns:
            List of dhash strings
        """
        # Compute horizontal gradients
        # Compare each pixel to its right neighbor
        diff = images[:, :, :, 1:] > images[:, :, :, :-1]

        # Flatten to get 64-bit hash
        bits = diff.reshape(diff.shape[0], -1)[:, :64]  # Take first 64 bits

        # Convert to hex strings
        hashes = []
        for bit_array in bits:
            # Convert boolean tensor to numpy
            bit_np = bit_array.cpu().numpy().astype(np.uint8)
            # Pack bits into bytes
            hash_int = int("".join(str(b) for b in bit_np), 2)
            hash_hex = f"{hash_int:016x}"
            hashes.append(hash_hex)

        return hashes

    def _compute_ahash_batch(self, images: "torch.Tensor") -> List[str]:  # type: ignore[name-defined]
        """
        Compute aHash for a batch of images on GPU.

        Args:
            images: Batch of preprocessed images (batch, 1, 8, 8)

        Returns:
            List of ahash strings
        """
        # Compute mean for each image
        means = images.mean(dim=(1, 2, 3), keepdim=True)

        # Compare each pixel to mean
        bits = (images > means).reshape(images.shape[0], -1)[:, :64]

        # Convert to hex strings
        hashes = []
        for bit_array in bits:
            bit_np = bit_array.cpu().numpy().astype(np.uint8)
            hash_int = int("".join(str(b) for b in bit_np), 2)
            hash_hex = f"{hash_int:016x}"
            hashes.append(hash_hex)

        return hashes

    def _compute_whash_batch(self, images: "torch.Tensor") -> List[str]:  # type: ignore[name-defined]
        """
        Compute wHash for a batch of images on GPU.

        Note: This uses DCT approximation instead of full DWT for GPU compatibility.

        Args:
            images: Batch of preprocessed images (batch, 1, 8, 8)

        Returns:
            List of whash strings
        """
        # For now, use simple frequency-based hash
        # TODO: Implement proper DWT on GPU using cupy or custom CUDA kernel

        # Use 2D FFT as approximation
        fft = self.torch.fft.fft2(images.squeeze(1))
        fft_abs = self.torch.abs(fft)

        # Take low-frequency components (top-left quadrant)
        low_freq = fft_abs[:, : images.shape[2] // 2, : images.shape[3] // 2]

        # Compare to median
        medians = low_freq.reshape(low_freq.shape[0], -1).median(dim=1, keepdim=True)[
            0
        ]
        medians = medians.view(-1, 1, 1)

        bits = (low_freq > medians).reshape(low_freq.shape[0], -1)[:, :64]

        # Convert to hex strings
        hashes = []
        for bit_array in bits:
            bit_np = bit_array.cpu().numpy().astype(np.uint8)
            hash_int = int("".join(str(b) for b in bit_np), 2)
            hash_hex = f"{hash_int:016x}"
            hashes.append(hash_hex)

        return hashes

    def compute_hashes_batch(
        self, image_paths: List[Path]
    ) -> List[Dict[str, Optional[str]]]:
        """
        Compute all hash types for a batch of images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of dicts with keys: dhash, ahash, whash
        """
        if not self.use_gpu:
            # Fallback to CPU version
            from vam_tools.analysis.perceptual_hash import combined_hash

            return [combined_hash(path) for path in image_paths]

        results = []

        try:
            # Process dhash (needs 9x9 for horizontal comparison)
            images_9 = self._load_and_preprocess(image_paths, target_size=9)
            if images_9 is not None:
                dhashes = self._compute_dhash_batch(images_9)
            else:
                dhashes = [None] * len(image_paths)

            # Process ahash and whash (need 8x8)
            images_8 = self._load_and_preprocess(image_paths, target_size=8)
            if images_8 is not None:
                ahashes = self._compute_ahash_batch(images_8)
                whashes = self._compute_whash_batch(images_8)
            else:
                ahashes = [None] * len(image_paths)
                whashes = [None] * len(image_paths)

            # Combine results
            for dhash, ahash, whash in zip(dhashes, ahashes, whashes):
                results.append({"dhash": dhash, "ahash": ahash, "whash": whash})

        except Exception as e:
            logger.error(f"GPU batch processing failed: {e}, falling back to CPU")
            # Fallback to CPU
            from vam_tools.analysis.perceptual_hash import combined_hash

            results = [combined_hash(path) for path in image_paths]

        return results

    def process_images(
        self, image_paths: List[Path]
    ) -> List[Dict[str, Optional[str]]]:
        """
        Process all images in batches.

        Args:
            image_paths: List of all image file paths

        Returns:
            List of hash dicts for each image
        """
        all_results = []

        # Process in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i : i + self.batch_size]
            batch_results = self.compute_hashes_batch(batch)
            all_results.extend(batch_results)

            if (i // self.batch_size) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(image_paths)} images")

        return all_results
