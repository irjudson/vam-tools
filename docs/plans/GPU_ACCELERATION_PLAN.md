# GPU Acceleration Plan for Lumina

## Executive Summary

Current performance: **~3ms per image** for hash computation (CPU)
Target with GPU: **10-50x speedup** for batch processing

## Current Bottlenecks

Based on profiling, the main bottlenecks are:

1. **Image I/O and Preprocessing** (60-70% of time)
   - Loading images from disk
   - Decoding JPEG/PNG
   - Resizing and color conversion

2. **Wavelet Transform (wHash)** (20-25% of time)
   - 2D Discrete Wavelet Transform
   - Most computationally expensive hash method

3. **Perceptual Hash Computation** (10-15% of time)
   - Pixel comparisons (dHash, aHash)
   - Less expensive but still benefits from parallelization

4. **Similarity Comparisons** (5% of time)
   - Pairwise hamming distance calculations
   - O(n²) complexity for large catalogs

## GPU Acceleration Opportunities

### Phase 1: Image Preprocessing (Highest Impact)

**Current**: PIL/Pillow (CPU-only)
**Proposed**: GPU-accelerated image decoding and preprocessing

#### Option A: NVIDIA DALI (Recommended)
- **Pros**:
  - Purpose-built for image processing pipelines
  - Very fast JPEG decoding on GPU
  - Excellent batch processing
  - Easy integration with PyTorch/NumPy
- **Cons**:
  - NVIDIA GPUs only
  - Additional dependency (~500MB)

```python
from nvidia.dali import pipeline_def, fn
from nvidia.dali.types import DALIDataType

@pipeline_def
def image_preprocessing_pipeline():
    jpegs, _ = fn.readers.file(file_root="/path/to/images")
    images = fn.decoders.image(jpegs, device="mixed")  # GPU decode
    images = fn.resize(images, size=[8, 8])  # GPU resize
    images = fn.color_space_conversion(images, image_type=types.RGB, output_type=types.GRAY)
    return images
```

#### Option B: OpenCV with CUDA
- **Pros**:
  - More flexible
  - Works with various GPUs (via OpenCL)
  - Familiar API
- **Cons**:
  - Requires compilation with CUDA support
  - Less optimized than DALI

#### Option C: TorchVision with CUDA
- **Pros**:
  - Easy to use if PyTorch is available
  - Good performance
  - Works with image tensors
- **Cons**:
  - Requires PyTorch (~1GB)
  - Overhead of tensor conversions

**Implementation Strategy**:
```python
class GPUImagePreprocessor:
    """GPU-accelerated image preprocessing."""

    def __init__(self, batch_size: int = 32, device: str = "cuda"):
        self.batch_size = batch_size
        self.device = device
        self.pipeline = self._build_pipeline()

    def preprocess_batch(self, image_paths: List[Path]) -> torch.Tensor:
        """Process a batch of images on GPU."""
        # Load and decode on GPU
        images = self._gpu_decode(image_paths)
        # Resize on GPU
        images = self._gpu_resize(images, (8, 8))
        # Convert to grayscale on GPU
        images = self._gpu_to_grayscale(images)
        return images
```

### Phase 2: Wavelet Transform Acceleration

**Current**: PyWavelets (CPU-only)
**Proposed**: GPU-accelerated DWT

#### Option A: CuPy + Custom DWT
```python
import cupy as cp
from cupyx.scipy import signal

def gpu_whash(image_gpu: cp.ndarray, hash_size: int = 8) -> str:
    """Compute wavelet hash on GPU using CuPy."""
    # Implement 2D DWT using GPU-accelerated FFT
    coeffs_ll = cp_dwt2d(image_gpu, mode='haar')
    median = cp.median(coeffs_ll)
    hash_bits = coeffs_ll > median
    return bits_to_hex(cp.asnumpy(hash_bits))
```

#### Option B: PyTorch FFT-based Approximation
- Use PyTorch's FFT operations on GPU
- Approximate DWT with DCT or FFT
- Trade slight accuracy for massive speed

### Phase 3: Batch Hash Computation

**Key Insight**: Process multiple images simultaneously on GPU

```python
class BatchHasher:
    """Compute hashes for multiple images in parallel on GPU."""

    def __init__(self, device: str = "cuda", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.preprocessor = GPUImagePreprocessor(batch_size, device)

    def compute_hashes_batch(
        self, image_paths: List[Path]
    ) -> List[Dict[str, str]]:
        """Compute all hash types for a batch of images on GPU."""
        hashes = []

        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i:i + self.batch_size]

            # Preprocess entire batch on GPU
            images_gpu = self.preprocessor.preprocess_batch(batch)

            # Compute all three hash types in parallel on GPU
            dhashes = self._batch_dhash_gpu(images_gpu)
            ahashes = self._batch_ahash_gpu(images_gpu)
            whashes = self._batch_whash_gpu(images_gpu)

            # Combine results
            for d, a, w in zip(dhashes, ahashes, whashes):
                hashes.append({"dhash": d, "ahash": a, "whash": w})

        return hashes
```

### Phase 4: GPU-Accelerated Similarity Search

**Current**: O(n²) CPU comparisons
**Proposed**: GPU-accelerated nearest neighbor search

#### Use FAISS for Fast Similarity Search
```python
import faiss

class GPUSimilaritySearcher:
    """Fast GPU-based similarity search using FAISS."""

    def __init__(self, hash_size: int = 64):
        self.hash_size = hash_size
        # Create GPU index for binary vectors (hashes)
        self.index = faiss.IndexBinaryFlat(hash_size)
        # Move to GPU
        self.gpu_index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(), 0, self.index
        )

    def find_similar(
        self, hashes: np.ndarray, threshold: int = 5
    ) -> List[Tuple[int, int]]:
        """Find all pairs of similar images using GPU."""
        # Add all hashes to index
        self.gpu_index.add(hashes)

        # Search for neighbors within threshold
        # Returns in milliseconds instead of seconds!
        distances, indices = self.gpu_index.search(hashes, k=100)

        # Filter by threshold
        similar_pairs = []
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            for dist, idx in zip(dists, idxs):
                if 0 < dist <= threshold and i < idx:
                    similar_pairs.append((i, idx))

        return similar_pairs
```

## Implementation Roadmap

### Phase 1 (Week 1): Detection and Fallback
```python
def detect_gpu_capability() -> Dict[str, Any]:
    """Detect available GPU and capabilities."""
    gpu_info = {
        "available": False,
        "device_name": None,
        "memory_gb": 0,
        "cuda_version": None,
        "backends": []
    }

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_info["backends"].append("cuda")
    except ImportError:
        pass

    # Check OpenCL
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        if platforms:
            gpu_info["backends"].append("opencl")
    except ImportError:
        pass

    return gpu_info
```

### Phase 2 (Week 2): Hybrid CPU/GPU Processing
```python
class HybridHasher:
    """Intelligent routing between CPU and GPU based on workload."""

    def __init__(self):
        self.gpu_info = detect_gpu_capability()
        self.use_gpu = self.gpu_info["available"]

        if self.use_gpu:
            self.gpu_hasher = BatchHasher()
        else:
            logger.info("GPU not available, using CPU")

    def compute_hashes(
        self, image_paths: List[Path], force_cpu: bool = False
    ) -> List[Dict[str, str]]:
        """Compute hashes using best available method."""
        # Small batch? Use CPU (overhead not worth it)
        if force_cpu or not self.use_gpu or len(image_paths) < 10:
            return self._compute_hashes_cpu(image_paths)

        # Large batch? Use GPU
        return self.gpu_hasher.compute_hashes_batch(image_paths)
```

### Phase 3 (Week 3): Configuration and Optimization
```python
class GPUConfig(BaseModel):
    """GPU acceleration configuration."""

    enabled: bool = True
    batch_size: int = 32  # Tunable based on GPU memory
    device_id: int = 0    # For multi-GPU systems
    memory_limit_gb: Optional[float] = None
    fallback_to_cpu: bool = True
    backends: List[str] = ["cuda", "opencl"]  # Preference order
```

Add to CLI:
```bash
# Enable GPU acceleration
vam-analyze /catalog --detect-duplicates --gpu

# Configure GPU settings
vam-analyze /catalog --detect-duplicates --gpu --gpu-batch-size 64

# Force CPU (for testing/comparison)
vam-analyze /catalog --detect-duplicates --no-gpu
```

## Expected Performance Improvements

### Single Image Processing
- **Current (CPU)**: 3ms per image
- **With GPU**: 3ms per image (no benefit due to overhead)

### Batch Processing (1000 images)
- **Current (CPU)**: 3000ms (3 seconds)
- **With GPU (Option A - DALI)**: 150-300ms (10-20x faster)
- **With GPU (Option B - OpenCV)**: 300-600ms (5-10x faster)

### Large Catalog (87,500 images - your use case!)
- **Current (CPU)**: ~262 seconds (~4.4 minutes)
- **With GPU**: ~13-26 seconds (10-20x faster)

### Similarity Search (N=10,000)
- **Current (CPU)**: O(n²) = ~5 minutes
- **With GPU (FAISS)**: <1 second (300x faster!)

## Dependencies

### Minimal (CPU fallback only)
```toml
# No changes needed
```

### GPU-Enabled (Recommended)
```toml
[project.optional-dependencies]
gpu = [
    "torch>=2.0.0",  # For GPU tensor operations
    "cupy-cuda12x>=12.0.0",  # For GPU-accelerated NumPy operations
    "faiss-gpu>=1.7.4",  # For fast similarity search
]
```

### Full GPU (Maximum performance)
```toml
[project.optional-dependencies]
gpu-full = [
    "torch>=2.0.0",
    "cupy-cuda12x>=12.0.0",
    "faiss-gpu>=1.7.4",
    "nvidia-dali-cuda120>=1.30.0",  # For fastest image preprocessing
]
```

## Testing Strategy

1. **Unit tests**: Verify GPU and CPU produce same results
2. **Performance benchmarks**: Compare GPU vs CPU speeds
3. **Memory tests**: Ensure GPU memory is managed properly
4. **Fallback tests**: Verify graceful degradation without GPU

```python
class TestGPUAcceleration:
    """Tests for GPU-accelerated processing."""

    def test_gpu_cpu_hash_equivalence(self):
        """GPU and CPU should produce identical hashes."""
        image = create_test_image()

        hash_cpu = dhash_cpu(image)
        hash_gpu = dhash_gpu(image)

        assert hash_cpu == hash_gpu

    @pytest.mark.benchmark
    def test_gpu_speedup(self):
        """Measure GPU speedup over CPU."""
        images = [create_test_image() for _ in range(100)]

        time_cpu = timeit(lambda: batch_hash_cpu(images))
        time_gpu = timeit(lambda: batch_hash_gpu(images))

        speedup = time_cpu / time_gpu
        assert speedup > 5  # At least 5x faster
```

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| GPU not available | Automatic CPU fallback |
| GPU memory overflow | Adaptive batch sizing |
| Different GPU vendors | Multi-backend support (CUDA/OpenCL) |
| Numerical differences | Validation tests ensuring equivalence |
| Increased dependencies | Make GPU support optional |

## Recommendation

**Start with Phase 1 + Phase 4**:
1. Implement GPU detection and graceful fallback
2. Implement FAISS-based similarity search (biggest win for large catalogs)
3. Monitor usage and user feedback
4. Add full batch processing if needed

For your use case (87,500 images):
- **Similarity search would benefit most** (5 min → <1 sec)
- **Hash computation** already fast enough (4 min total)
- **Focus GPU effort on similarity comparisons first**
