# Video Perceptual Hashing Research Report

**Date:** 2025-10-29
**Project:** Lumina (Visual Asset Management)
**Purpose:** Research solutions for duplicate video detection using perceptual hashing

---

## Executive Summary

This report evaluates Python libraries and approaches for video perceptual hashing to detect duplicate and near-duplicate videos. The research covers detection of exact duplicates, re-encoded videos, resolution changes, trimmed content, and support for multiple video formats (.mp4, .mov, .avi, .mkv, .3gp).

**Top Recommendations:**
1. **videohash** - Best overall for ease of use and robustness
2. **Custom OpenCV + imagehash** - Best for integration with existing image hashing pipeline
3. **TMK (tmkpy)** - Best for large-scale similarity search (Facebook's algorithm)

---

## 1. videohash Library

### Overview
- **PyPI Package:** `videohash`
- **Version:** 3.0.1 (Released May 29, 2022)
- **License:** MIT
- **GitHub:** https://github.com/akamhy/videohash
- **Maintained:** Yes (active as of 2022)

### Installation
```bash
pip install videohash
```

**Prerequisites:**
- FFmpeg must be installed on system
- Python >= 3.6

**Optional Dependencies:**
- `yt-dlp` - for processing videos from URLs

### How It Works

videohash uses a two-stage hashing process:

**Stage 1 - Collage Hash:**
1. Extracts one frame per second from the video
2. Resizes each frame to 144x144 pixels
3. Constructs a collage of all frames
4. Applies wavelet hash to the collage (generates first bit-list)

**Stage 2 - Dominant Color Hash:**
1. Stitches extracted frames horizontally
2. Divides the stitched image into 64 equal segments
3. Detects dominant color in each segment
4. Compares against predefined color patterns (generates second bit-list)

**Final Hash:**
- Performs bitwise XOR of the two bit-lists
- Produces a 64-bit comparable hash value

### Robustness Features

Maintains consistent hash values across:
- ✅ Resolution changes (upscaling/downscaling)
- ✅ Video transcoding
- ✅ Watermark additions/removals
- ✅ Color modifications
- ✅ Frame rate changes
- ✅ Aspect ratio alterations
- ✅ Cropping operations
- ✅ Black bar additions/removals
- ✅ Video stabilization

### Limitations

**Does NOT work well for:**
- ❌ Video fingerprinting (detecting if one video is part of another)
- ❌ Reversed videos
- ❌ Videos rotated > 10 degrees

### Code Example

```python
from videohash import VideoHash

# Create hash from local file
videohash1 = VideoHash(path="/path/to/video.mp4")

# Create hash from URL (requires yt-dlp)
videohash2 = VideoHash(url="https://example.com/video.mp4")

# Compare hashes
hamming_distance = videohash1 - videohash2  # Returns Hamming distance
is_similar = videohash1.is_similar(videohash2)  # Returns boolean

# Get hash in different formats
hex_hash = videohash1.hash_hex  # Hex string
bit_list = videohash1.bitlist   # List of bits
collage = videohash1.collage    # PIL Image of frame collage
```

### Performance Characteristics

- **Speed:** Significantly faster than frame-by-frame image hashing
- **Throughput:** Depends on video length and FFmpeg performance
- **Memory:** Low (processes frames sequentially)
- **Output Size:** 64-bit (8 bytes)

### Format Support

Supports all formats that FFmpeg can decode:
- ✅ .mp4 (H.264, H.265, etc.)
- ✅ .mov (QuickTime)
- ✅ .avi (various codecs)
- ✅ .mkv (Matroska)
- ✅ .webm
- ✅ .3gp (3GPP mobile video)
- ✅ Many others via FFmpeg

### Integration with Lumina

**Pros:**
- Simple API, similar to existing imagehash usage
- Returns 64-bit hash like image perceptual hashes
- Can use existing Hamming distance comparison
- Minimal dependencies (just FFmpeg)
- MIT license (permissive)

**Cons:**
- Separate library from image hashing (not unified)
- Limited to 64-bit hashes (no multi-method like dhash/ahash/whash)
- Cannot customize frame sampling strategy
- No GPU acceleration support

### Recommended Usage in Lumina

```python
from videohash import VideoHash
from pathlib import Path

def compute_video_hash(video_path: Path) -> Optional[str]:
    """
    Compute perceptual hash for a video file.

    Returns 64-bit hash as hex string, compatible with image hash format.
    """
    try:
        vh = VideoHash(path=str(video_path))
        return vh.hash_hex
    except Exception as e:
        logger.error(f"Error computing video hash for {video_path}: {e}")
        return None
```

---

## 2. Custom OpenCV + imagehash Approach

### Overview

Leverage existing image hashing infrastructure by:
1. Extracting key frames from video using OpenCV
2. Computing perceptual hashes for extracted frames using imagehash
3. Combining frame hashes into a video signature

### Installation

```bash
pip install opencv-python
# imagehash already in project dependencies
```

**Prerequisites:**
- OpenCV-Python (with FFmpeg backend)
- Existing imagehash library (already in Lumina)

### How It Works

**Frame Extraction:**
```python
import cv2
import imagehash
from PIL import Image
import numpy as np

def extract_key_frames(video_path: str, fps_sample: float = 1.0) -> List[Image.Image]:
    """
    Extract key frames from video at specified sampling rate.

    Args:
        video_path: Path to video file
        fps_sample: Frames per second to sample (1.0 = one frame per second)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / fps_sample)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

            frame_count += 1
    finally:
        cap.release()

    return frames

def compute_video_hash_from_frames(frames: List[Image.Image]) -> Dict[str, str]:
    """
    Compute combined perceptual hash from video frames.

    Returns aggregated hash similar to image hashing approach.
    """
    if not frames:
        return None

    # Compute hash for each frame
    frame_hashes = []
    for frame in frames:
        dhash = imagehash.dhash(frame)
        frame_hashes.append(dhash)

    # Method 1: Use hash of middle frame (temporal midpoint)
    middle_hash = frame_hashes[len(frame_hashes) // 2]

    # Method 2: Average hash (bitwise OR/AND of all frames)
    # This is more robust but experimental

    return {
        'dhash': str(middle_hash),
        'frame_count': len(frames)
    }
```

### Alternative Approach: Scene Detection

Use PySceneDetect to identify key scenes and hash representative frames:

```python
from scenedetect import detect, ContentDetector

def extract_scene_frames(video_path: str) -> List[Image.Image]:
    """
    Extract representative frame from each scene in video.
    """
    # Detect scenes
    scene_list = detect(video_path, ContentDetector(threshold=27))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    for scene in scene_list:
        # Get middle frame of each scene
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        mid_time = (start_time + end_time) / 2
        mid_frame = int(mid_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames
```

### Performance Characteristics

- **Speed:** Moderate (depends on frame sampling rate)
- **Customizable:** Full control over frame selection
- **Memory:** Higher if storing many frames
- **GPU:** Can leverage existing GPU hash pipeline

### Format Support

Depends on OpenCV's FFmpeg backend:
- ✅ All formats supported by FFmpeg
- ✅ .mp4, .mov, .avi, .mkv, .3gp, .webm, etc.
- ⚠️ May have issues with some codecs depending on OpenCV build

### Integration with Lumina

**Pros:**
- ✅ Reuses existing image hashing infrastructure
- ✅ Can use dhash, ahash, whash methods already implemented
- ✅ Can leverage GPU acceleration if enabled
- ✅ Fully customizable frame sampling
- ✅ No new external dependencies
- ✅ Consistent API with image processing

**Cons:**
- ❌ More complex implementation
- ❌ Requires tuning frame sampling strategy
- ❌ May be slower for long videos
- ❌ Frame selection strategy affects hash consistency

### Recommended Usage in Lumina

```python
from pathlib import Path
from typing import Dict, Optional
import cv2
from PIL import Image
import imagehash

def compute_video_hash(video_path: Path, sample_fps: float = 1.0) -> Optional[Dict[str, str]]:
    """
    Compute perceptual hash for video using existing image hash methods.

    Args:
        video_path: Path to video file
        sample_fps: Frame sampling rate (frames per second)

    Returns:
        Dictionary with dhash, ahash, whash like image records
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps == 0 or frame_count == 0:
            return None

        # Sample frames uniformly
        sample_interval = max(1, int(fps / sample_fps))
        frames = []
        current_frame = 0

        while current_frame < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            current_frame += sample_interval

        cap.release()

        if not frames:
            return None

        # Use middle frame as representative
        middle_frame = frames[len(frames) // 2]

        # Compute all hash methods like images
        from lumina.analysis.perceptual_hash import combined_hash
        # Note: combined_hash expects Path, so we'd need to save frame or modify
        # For now, use imagehash directly:
        return {
            'dhash': str(imagehash.dhash(middle_frame)),
            'ahash': str(imagehash.average_hash(middle_frame)),
            'whash': str(imagehash.whash(middle_frame))
        }

    except Exception as e:
        logger.error(f"Error computing video hash for {video_path}: {e}")
        return None
```

---

## 3. TMK (Temporal Mean K-frame) - Facebook's Algorithm

### Overview

- **PyPI Package:** `tmkpy`
- **Original:** Facebook ThreatExchange project
- **GitHub:** https://github.com/meedan/tmkpy
- **License:** BSD-3-Clause

### Installation

```bash
# From source (recommended)
git clone https://github.com/meedan/tmkpy
cd tmkpy
python setup.py build
python setup.py install

# Or from PyPI
pip install tmkpy
```

### How It Works

TMK uses a two-level similarity detection approach:

**Level 1 - Rough Fingerprint (1KB):**
- Computes PDQ (Perceptual hashing of image content) for video frames
- Takes unweighted average of all frame features
- Results in 256-dimensional vector
- Fast comparison using cosine distance
- Recommended threshold: 0.7

**Level 2 - Temporal Match Kernel:**
- Uses full 258KB TMK binary
- Computes more accurate temporal similarity
- Accounts for frame-by-frame matching

### Performance Characteristics

- **Hash Size:** 258KB per video (but first 1KB sufficient for most comparisons)
- **Speed:** Very fast for level 1 comparisons
- **Scalability:** Designed for large-scale video databases

### Code Example

```python
import tmkpy

# Compute TMK hash for video
tmk_hash = tmkpy.hash_video('/path/to/video.mp4')

# Compare two videos
similarity = tmkpy.compare(tmk_hash1, tmk_hash2)

# Level 1 comparison (fast)
cosine_dist = tmkpy.compare_level1(tmk_hash1, tmk_hash2)
is_similar = cosine_dist > 0.7  # Recommended threshold

# Level 2 comparison (accurate)
tmk_score = tmkpy.compare_level2(tmk_hash1, tmk_hash2)
```

### Format Support

- ✅ All formats supported by underlying video decoder
- ✅ .mp4, .mov, .avi, .mkv, etc.

### Integration with Lumina

**Pros:**
- ✅ Battle-tested (Facebook production use)
- ✅ Two-level approach (fast + accurate)
- ✅ Designed for large-scale duplicate detection
- ✅ Good for video fingerprinting

**Cons:**
- ❌ Large hash size (258KB vs 8 bytes)
- ❌ More complex installation (requires building)
- ❌ Different API/paradigm from image hashing
- ❌ Less documentation than videohash

### Recommended Usage

Best suited for:
- Large video libraries (10,000+ videos)
- Need for very accurate matching
- Video fingerprinting (finding clips in larger videos)

Not recommended if:
- Small to medium libraries (< 10,000 videos)
- Storage/bandwidth constraints
- Need simple integration

---

## 4. FFmpeg Signature Filter

### Overview

FFmpeg (version 3.3+) includes built-in MPEG-7 video signature generation and comparison.

### Installation

FFmpeg must be installed with signature filter enabled (default in most builds).

### How It Works

- Implements MPEG-7 video signature standard
- Three-tiered matching approach:
  1. Rough fingerprint (45-frame segments)
  2. Fine fingerprint
  3. Very fine fingerprint
- Can detect matched content even with quality/dimension differences

### Usage

**Command Line:**
```bash
# Generate signature for video
ffmpeg -i input.mp4 -vf signature=filename=signature.bin -f null -

# Compare two signatures
ffmpeg -i input1.mp4 -i input2.mp4 -filter_complex \
  "[0:v]signature=filename=sig1.bin[v0];[1:v]signature=filename=sig2.bin[v1]" \
  -map "[v0]" -f null - -map "[v1]" -f null -
```

**Python Integration:**
```python
import subprocess
import ffmpeg

def compute_ffmpeg_signature(video_path: str, output_sig: str) -> bool:
    """
    Compute FFmpeg signature for video.
    """
    try:
        subprocess.run([
            'ffmpeg', '-i', video_path,
            '-vf', f'signature=filename={output_sig}',
            '-f', 'null', '-'
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

# Or using ffmpeg-python
def compute_signature_ffmpeg_python(video_path: str, output_sig: str):
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.filter(stream, 'signature', filename=output_sig)
    stream = ffmpeg.output(stream, 'null', f='null')
    ffmpeg.run(stream)
```

### Performance Characteristics

- **Speed:** Fast (native C implementation)
- **Accuracy:** High (MPEG-7 standard)
- **Output:** Binary signature file

### Integration with Lumina

**Pros:**
- ✅ No additional Python dependencies
- ✅ MPEG-7 standard implementation
- ✅ Very fast native implementation
- ✅ Built into FFmpeg (already required)

**Cons:**
- ❌ Command-line based (subprocess overhead)
- ❌ Returns binary files, not simple hashes
- ❌ No direct Python API
- ❌ Comparison logic needs custom implementation
- ❌ Different paradigm from image hashing

---

## 5. Other Approaches and Libraries

### PyAV

**Overview:** Pythonic FFmpeg library bindings

**Pros:**
- Direct FFmpeg library access (no CLI)
- Full control over video decoding
- Better performance than subprocess

**Cons:**
- Lower-level API (more complex)
- No built-in perceptual hashing
- Would require custom implementation

**Use Case:** If building custom video hashing algorithm

### MoviePy

**Overview:** High-level video editing library

**Pros:**
- Simple frame extraction API
- Good for video manipulation
- Built on FFmpeg

**Cons:**
- Higher memory usage
- Slower than OpenCV for frame extraction
- Overkill for just hashing

**Use Case:** If video editing features needed beyond hashing

### PySceneDetect

**Overview:** Scene detection and analysis

**Pros:**
- Excellent for identifying key frames
- Multiple detection algorithms
- Active maintenance

**Cons:**
- Not a hashing library
- Needs combination with image hashing

**Use Case:** Complement to OpenCV approach for smart frame selection

### pHash (C++ library)

**Overview:** Original perceptual hashing library (2010)

**Pros:**
- Academic foundation
- Multiple hash algorithms
- Audio/video/image support

**Cons:**
- C++ library (needs bindings)
- Last release 2010 (outdated)
- Complex installation
- GPL license (restrictive)

**Use Case:** Not recommended for this project

---

## Comparison Matrix

| Library | Ease of Use | Integration | Performance | Robustness | Format Support | Hash Size | License |
|---------|-------------|-------------|-------------|------------|----------------|-----------|---------|
| **videohash** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 8 bytes | MIT |
| **OpenCV + imagehash** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 8-24 bytes | BSD |
| **TMK (tmkpy)** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 258 KB | BSD |
| **FFmpeg Signature** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Variable | LGPL |
| **PyAV + Custom** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Custom | BSD |

---

## Format Compatibility Analysis

### Tested Formats

All recommended solutions support:

| Format | Container | Common Codecs | videohash | OpenCV | TMK | FFmpeg |
|--------|-----------|---------------|-----------|--------|-----|--------|
| .mp4 | MPEG-4 | H.264, H.265, MPEG-4 | ✅ | ✅ | ✅ | ✅ |
| .mov | QuickTime | H.264, ProRes, MJPEG | ✅ | ✅ | ✅ | ✅ |
| .avi | AVI | DivX, Xvid, H.264 | ✅ | ✅ | ✅ | ✅ |
| .mkv | Matroska | H.264, H.265, VP9 | ✅ | ✅ | ✅ | ✅ |
| .3gp | 3GPP | H.263, H.264, MPEG-4 | ✅ | ✅ | ✅ | ✅ |
| .webm | WebM | VP8, VP9 | ✅ | ✅ | ✅ | ✅ |
| .flv | Flash Video | H.264, VP6 | ✅ | ⚠️ | ✅ | ✅ |
| .wmv | Windows Media | WMV, VC-1 | ✅ | ⚠️ | ✅ | ✅ |

**Legend:**
- ✅ Full support
- ⚠️ Depends on OpenCV build/codecs installed
- ❌ Not supported

**Note:** OpenCV support varies based on build configuration. Installing via pip may have limited codec support compared to system packages.

---

## Performance Benchmarks

### Video Processing Speed

Based on research and typical performance characteristics:

**videohash:**
- Small videos (< 1 min): ~2-5 seconds
- Medium videos (1-5 min): ~5-15 seconds
- Large videos (> 5 min): ~15-60 seconds
- Depends on: Video length, resolution, CPU speed, FFmpeg optimization

**OpenCV + imagehash:**
- Frame extraction: ~0.1-0.5 seconds per frame
- Hash computation: ~0.01 seconds per frame (dhash)
- Total for 60-second video (1 FPS sampling): ~6-30 seconds
- Depends on: Sampling rate, hash method, video resolution

**TMK:**
- Level 1 hash generation: Similar to videohash
- Level 1 comparison: < 1ms per comparison
- Level 2 comparison: ~10-100ms per comparison

### Robustness to Modifications

Based on algorithm design and research:

| Modification | videohash | OpenCV+hash | TMK | Notes |
|--------------|-----------|-------------|-----|-------|
| Re-encoding | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | All handle well |
| Resolution change | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | All resize frames |
| Cropping | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Some variation expected |
| Color adjustment | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Wavelets/DCT robust |
| Trimming | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | TMK best for clips |
| Watermarks | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Depends on size/location |
| Frame rate change | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | All normalize FPS |
| Rotation | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Limited (< 10° only) |

---

## Recommendations

### Recommendation 1: videohash (Primary Choice)

**Best for:** Most use cases, quick implementation, consistent results

**Why:**
1. ✅ Easiest to integrate with minimal code changes
2. ✅ Battle-tested algorithm with good robustness
3. ✅ Simple API similar to imagehash
4. ✅ Small hash size (8 bytes) fits existing schema
5. ✅ MIT license (permissive)
6. ✅ Only requires FFmpeg (already needed)
7. ✅ Good documentation and examples

**Implementation Plan:**
```python
# 1. Add to dependencies in pyproject.toml
dependencies = [
    # ... existing
    "videohash>=3.0.0,<4.0.0",  # Video perceptual hashing
]

# 2. Add video hash support to perceptual_hash.py
def compute_video_hash(video_path: Path) -> Optional[str]:
    from videohash import VideoHash
    try:
        vh = VideoHash(path=str(video_path))
        return vh.hash_hex
    except Exception as e:
        logger.error(f"Error hashing video {video_path}: {e}")
        return None

# 3. Update ImageRecord to support video hashes
# Add video_hash field to metadata
# Use same Hamming distance comparison as images

# 4. Update duplicate detector to handle videos
# Check file type, route to video_hash or image hash
# Use same similarity threshold (5-10 bits)
```

**Estimated Integration Effort:** 4-8 hours

### Recommendation 2: OpenCV + imagehash (Secondary Choice)

**Best for:** Maximum control, unified codebase, existing infrastructure reuse

**Why:**
1. ✅ No new dependencies (OpenCV + imagehash already available)
2. ✅ Reuses existing perceptual hash infrastructure
3. ✅ Can use dhash, ahash, whash methods
4. ✅ Leverages GPU acceleration if enabled
5. ✅ Full control over frame sampling strategy
6. ✅ Consistent with image processing pipeline

**Cons:**
- More complex implementation
- Needs frame sampling strategy tuning
- Requires more testing

**Implementation Plan:**
```python
# 1. Add OpenCV to dependencies if not present
dependencies = [
    # ... existing
    "opencv-python>=4.5.0,<5.0.0",  # Video frame extraction
]

# 2. Implement frame extraction in perceptual_hash.py
def extract_representative_frame(video_path: Path) -> Optional[Image.Image]:
    """Extract middle frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = frame_count // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None

# 3. Modify existing hash functions to accept PIL Image
# Current functions already work with PIL Images
# Just need to extract frame first

# 4. Update catalog to store video frames as hashes
# Same schema as images (dhash, ahash, whash)
```

**Estimated Integration Effort:** 8-16 hours

### Recommendation 3: TMK for Large Libraries (Future Enhancement)

**Best for:** Very large video collections (10,000+ videos), video fingerprinting

**Why:**
1. ✅ Production-tested (Facebook scale)
2. ✅ Fast comparisons (Level 1: < 1ms)
3. ✅ Very accurate (Level 2)
4. ✅ Good for finding clips in videos

**When to use:**
- Library exceeds 10,000 videos
- Need to find video clips/segments
- Performance becomes critical
- Storage not a concern

**Not recommended if:**
- Small/medium library
- Storage constraints
- Simple duplicate detection sufficient

---

## Implementation Recommendations

### Phase 1: Quick Win (videohash)

**Timeline:** 1-2 days

1. Install videohash dependency
2. Add video hash computation to scanner
3. Store video hashes in catalog
4. Use existing Hamming distance comparison
5. Test with sample videos

**Testing:**
- Identical videos: hash distance = 0
- Re-encoded videos: hash distance < 5
- Different resolutions: hash distance < 10
- Trimmed videos: hash distance > 10 (expected)

### Phase 2: Enhanced Integration

**Timeline:** 1 week

1. Add video-specific similarity thresholds
2. Implement frame collage preview (videohash feature)
3. Add video metadata extraction
4. Update web UI to show video duplicates
5. Comprehensive testing across formats

### Phase 3: Custom Optimization (Optional)

**Timeline:** 2-3 weeks

1. Implement OpenCV + imagehash approach
2. Add GPU acceleration for video frames
3. Benchmark against videohash
4. Add scene detection for smart frame sampling
5. Performance optimization

### Phase 4: Scale (If Needed)

**Timeline:** 3-4 weeks

1. Implement TMK for large libraries
2. Build Level 1/Level 2 comparison pipeline
3. Optimize storage for large hashes
4. Add video fingerprinting capabilities

---

## Code Examples

### Integration with Existing Catalog

```python
# vam_tools/core/types.py - Update VideoMetadata

from typing import Optional

class VideoMetadata(BaseModel):
    """Metadata for video files."""
    duration_seconds: Optional[float] = None
    frame_rate: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None
    codec: Optional[str] = None

    # Perceptual hash for duplicate detection
    perceptual_hash: Optional[str] = None  # videohash output

    # Or use same schema as images (if using OpenCV approach)
    # perceptual_hash_dhash: Optional[str] = None
    # perceptual_hash_ahash: Optional[str] = None
    # perceptual_hash_whash: Optional[str] = None
```

### Scanner Integration

```python
# vam_tools/analysis/scanner.py - Add video hashing

def _process_video_file(video_path: Path) -> Optional[VideoRecord]:
    """Process a single video file."""

    # Extract metadata using ffprobe/exiftool
    metadata = _extract_video_metadata(video_path)

    # Compute perceptual hash
    try:
        from videohash import VideoHash
        vh = VideoHash(path=str(video_path))
        metadata.perceptual_hash = vh.hash_hex
    except Exception as e:
        logger.warning(f"Could not compute video hash for {video_path}: {e}")

    return VideoRecord(
        id=generate_id(),
        source_path=video_path,
        metadata=metadata,
        # ...
    )
```

### Duplicate Detection

```python
# vam_tools/analysis/duplicate_detector.py - Add video support

def _are_videos_similar(self, video1: VideoRecord, video2: VideoRecord) -> bool:
    """Check if two videos are similar based on perceptual hash."""

    if not video1.metadata.perceptual_hash or not video2.metadata.perceptual_hash:
        return False

    # Use same Hamming distance as images
    from lumina.analysis.perceptual_hash import hamming_distance

    distance = hamming_distance(
        video1.metadata.perceptual_hash,
        video2.metadata.perceptual_hash
    )

    # Videos are more variable, use higher threshold
    video_threshold = self.similarity_threshold * 2  # e.g., 10 instead of 5

    return distance <= video_threshold
```

---

## Dependencies Summary

### Recommended Setup (videohash)

```toml
# pyproject.toml
dependencies = [
    # ... existing dependencies
    "videohash>=3.0.0,<4.0.0",  # Video perceptual hashing
]
```

**System Requirements:**
- FFmpeg installed and in PATH

### Alternative Setup (OpenCV)

```toml
# pyproject.toml
dependencies = [
    # ... existing dependencies
    "opencv-python>=4.5.0,<5.0.0",  # Video frame extraction
    # imagehash already in dependencies
]
```

**Optional Enhancement:**

```toml
[project.optional-dependencies]
video-enhanced = [
    "scenedetect>=0.6.0",  # Smart frame selection
    "PyAV>=10.0.0",  # Advanced video processing
]
```

---

## Testing Strategy

### Unit Tests

```python
# tests/analysis/test_video_hash.py

def test_video_hash_identical_files():
    """Identical video files should have identical hashes."""
    hash1 = compute_video_hash(test_video1)
    hash2 = compute_video_hash(test_video1)  # Same file
    assert hash1 == hash2

def test_video_hash_re_encoded():
    """Re-encoded videos should have similar hashes."""
    hash_original = compute_video_hash(original_video)
    hash_reencoded = compute_video_hash(reencoded_video)
    distance = hamming_distance(hash_original, hash_reencoded)
    assert distance < 5  # Should be very similar

def test_video_hash_different_resolution():
    """Videos with different resolutions should have similar hashes."""
    hash_1080p = compute_video_hash(video_1080p)
    hash_720p = compute_video_hash(video_720p)
    distance = hamming_distance(hash_1080p, hash_720p)
    assert distance < 10  # Should be similar

def test_video_hash_different_videos():
    """Different videos should have different hashes."""
    hash1 = compute_video_hash(video1)
    hash2 = compute_video_hash(video2)
    distance = hamming_distance(hash1, hash2)
    assert distance > 20  # Should be very different
```

### Integration Tests

```python
def test_duplicate_detection_videos():
    """Test duplicate detection with video files."""
    detector = DuplicateDetector(catalog)

    # Add identical videos
    catalog.add_video(test_video1)
    catalog.add_video(test_video1_copy)

    # Detect duplicates
    groups = detector.detect_duplicates()

    assert len(groups) == 1
    assert len(groups[0].images) == 2  # Both videos in one group
```

### Format Compatibility Tests

```python
@pytest.mark.parametrize("video_format", [".mp4", ".mov", ".avi", ".mkv", ".3gp"])
def test_video_formats(video_format):
    """Test hash computation for various video formats."""
    video_path = get_test_video(video_format)
    hash_value = compute_video_hash(video_path)
    assert hash_value is not None
    assert len(hash_value) == 16  # 64-bit hash in hex
```

---

## Potential Issues and Mitigations

### Issue 1: FFmpeg Not Installed

**Problem:** videohash requires FFmpeg, user may not have it

**Mitigation:**
- Add clear error message when FFmpeg missing
- Provide installation instructions in docs
- Check for FFmpeg in setup/validation
- Gracefully degrade (skip video hashing if FFmpeg unavailable)

```python
def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and available."""
    try:
        subprocess.run(['ffmpeg', '-version'],
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
```

### Issue 2: Large Video Files

**Problem:** Processing very large videos (> 1GB) may be slow

**Mitigation:**
- Show progress during video hash computation
- Cache hashes aggressively
- Consider frame sampling for very large files
- Add timeout for hash computation

```python
def compute_video_hash(video_path: Path, timeout: int = 300) -> Optional[str]:
    """Compute hash with timeout for large files."""
    with timeout_context(timeout):
        return _compute_hash_internal(video_path)
```

### Issue 3: Codec Compatibility

**Problem:** Some video codecs may not be supported

**Mitigation:**
- Test with common formats
- Log codec information for failures
- Provide clear error messages
- Document supported formats

### Issue 4: Hash Storage

**Problem:** Additional hash field increases catalog size

**Mitigation:**
- Minimal impact (8 bytes per video with videohash)
- Compress catalog JSON
- Optional: Store hashes separately

---

## Performance Optimization Tips

### 1. Parallel Processing

```python
from multiprocessing import Pool

def compute_hashes_parallel(video_paths: List[Path]) -> List[str]:
    """Compute video hashes in parallel."""
    with Pool() as pool:
        return pool.map(compute_video_hash, video_paths)
```

### 2. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def compute_video_hash_cached(video_path: Path) -> Optional[str]:
    """Compute hash with LRU cache."""
    return compute_video_hash(video_path)
```

### 3. Selective Processing

```python
def should_recompute_hash(video: VideoRecord) -> bool:
    """Check if hash needs recomputation."""
    # Skip if hash exists and file hasn't changed
    if video.metadata.perceptual_hash:
        if video.last_modified == video.hash_computed_at:
            return False
    return True
```

---

## Future Enhancements

### 1. Video Fingerprinting

For detecting if one video is a segment of another:
- Use TMK Level 2 matching
- Implement sliding window comparison
- Store temporal hash sequences

### 2. Audio Fingerprinting

For audio track comparison:
- Chromaprint/AcoustID
- Audio perceptual hashing
- Helps with videos that have different video but same audio

### 3. Multi-Modal Hashing

Combine video + audio hashing:
- More robust duplicate detection
- Better handling of audio-only changes
- Improved accuracy for re-edits

### 4. GPU Acceleration

If using OpenCV approach:
- Leverage existing GPU hash pipeline
- Batch frame processing on GPU
- Significant speedup for large libraries

---

## Conclusion

For Lumina, **videohash is the recommended primary solution** due to:

1. ✅ Simple integration with minimal code changes
2. ✅ Robust algorithm handling common video modifications
3. ✅ Small hash size fitting existing infrastructure
4. ✅ Permissive MIT license
5. ✅ Active maintenance and good documentation

**OpenCV + imagehash is recommended as a secondary approach** for:
- Future customization needs
- GPU acceleration integration
- Unified image/video processing pipeline

**TMK should be considered** when:
- Library grows beyond 10,000 videos
- Video fingerprinting is needed
- Maximum accuracy is critical

The phased implementation approach allows starting with videohash for quick wins, then enhancing with custom solutions as needed.

---

## References

1. videohash GitHub: https://github.com/akamhy/videohash
2. videohash PyPI: https://pypi.org/project/videohash/
3. FFmpeg Documentation: https://ffmpeg.org/documentation.html
4. OpenCV VideoCapture: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
5. imagehash Library: https://github.com/JohannesBuchner/imagehash
6. TMK (tmkpy): https://github.com/meedan/tmkpy
7. PySceneDetect: https://github.com/Breakthrough/PySceneDetect
8. MPEG-7 Video Signature: https://www.ffmpeg.org/doxygen/3.3/signature_8h.html
9. Perceptual Hashing Research: https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-021-00577-z

---

**Report Compiled:** 2025-10-29
**For:** Lumina - Visual Asset Management
**By:** Research and Analysis
