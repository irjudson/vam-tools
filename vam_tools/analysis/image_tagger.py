"""
AI-based image tagging using local models.

This module provides automatic image tagging using locally-run AI models.
Two backends are supported:
- OpenCLIP: Fast batch inference using CLIP-style zero-shot classification
- Ollama: Vision language models (LLaVA, Qwen-VL) for detailed understanding

All inference runs locally - no data leaves your machine.

Example:
    Tag images with OpenCLIP (fast, batch-capable):
        >>> tagger = ImageTagger(backend="openclip")
        >>> tags = tagger.tag_image("/path/to/image.jpg")
        >>> print(tags)
        [("dogs", 0.85), ("outdoor", 0.72), ("daylight", 0.68)]

    Tag with Ollama/LLaVA (more accurate, slower):
        >>> tagger = ImageTagger(backend="ollama", model="llava")
        >>> tags = tagger.tag_image("/path/to/image.jpg")
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from vam_tools.analysis.tag_taxonomy import TagTaxonomy

logger = logging.getLogger(__name__)


@dataclass
class TagResult:
    """Result of tagging an image."""

    tag_name: str
    confidence: float  # Combined/final confidence
    category: str
    source: str  # "openclip", "ollama", or "combined"
    openclip_confidence: Optional[float] = None  # Confidence from OpenCLIP (if used)
    ollama_confidence: Optional[float] = None  # Confidence from Ollama (if used)


class TaggerBackend(ABC):
    """Abstract base class for tagger backends."""

    @abstractmethod
    def tag_image(
        self,
        image_path: Path,
        tag_names: List[str],
        threshold: float = 0.25,
        max_tags: int = 10,
    ) -> List[Tuple[str, float]]:
        """Tag a single image.

        Args:
            image_path: Path to the image file
            tag_names: List of possible tag names to match against
            threshold: Minimum confidence threshold (0.0 to 1.0)
            max_tags: Maximum number of tags to return

        Returns:
            List of (tag_name, confidence) tuples, sorted by confidence descending
        """
        pass

    @abstractmethod
    def tag_batch(
        self,
        image_paths: List[Path],
        tag_names: List[str],
        threshold: float = 0.25,
        max_tags: int = 10,
    ) -> Dict[Path, List[Tuple[str, float]]]:
        """Tag multiple images in batch.

        Args:
            image_paths: List of paths to image files
            tag_names: List of possible tag names to match against
            threshold: Minimum confidence threshold
            max_tags: Maximum tags per image

        Returns:
            Dictionary mapping image paths to tag results
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass


class OpenCLIPBackend(TaggerBackend):
    """OpenCLIP-based zero-shot image classification backend.

    Uses CLIP to compute similarity between images and text descriptions
    of tags. Fast and efficient for batch processing.
    """

    # Available model configurations: (architecture, pretrained weights, embedding_dim)
    MODELS = {
        "ViT-B-32": ("ViT-B-32", "laion2b_s34b_b79k", 512),  # Fast, good quality
        "ViT-L-14": ("ViT-L-14", "laion2b_s32b_b82k", 768),  # Slower, better quality
        "ViT-H-14": ("ViT-H-14", "laion2b_s32b_b79k", 1024),  # Highest quality
    }

    # Target embedding dimension for database storage
    TARGET_EMBEDDING_DIM = 768

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        device: Optional[str] = None,
    ) -> None:
        """Initialize OpenCLIP backend.

        Args:
            model_name: Model to use (ViT-B-32, ViT-L-14, ViT-H-14)
            device: Device to use (cuda, cpu, mps). Auto-detected if None.
        """
        self.model_name = model_name
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._text_embeddings: Any = None
        self._tag_names: Optional[List[str]] = None

        # Determine device
        if device is None:
            self._device = self._detect_device()
        else:
            self._device = device

        logger.info(f"OpenCLIP backend using device: {self._device}")

    def _detect_device(self) -> str:
        """Detect best available device."""
        try:
            import torch

            if torch.cuda.is_available():
                # Test CUDA actually works
                try:
                    torch.zeros(1, device="cuda")
                    return "cuda"
                except Exception:
                    pass
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def embedding_dim(self) -> int:
        """Return the native embedding dimension for the current model."""
        _, _, dim = self.MODELS.get(self.model_name, self.MODELS["ViT-B-32"])
        return dim

    # RAW formats that need special handling
    RAW_FORMATS = {
        ".arw",
        ".cr2",
        ".cr3",
        ".nef",
        ".dng",
        ".orf",
        ".rw2",
        ".pef",
        ".srw",
        ".raf",
        ".raw",
    }

    def _load_image(self, path: Path) -> Optional[Image.Image]:
        """Load an image, with RAW format support.

        Args:
            path: Path to the image file

        Returns:
            PIL Image in RGB mode, or None if loading failed
        """
        suffix = path.suffix.lower()

        # Handle RAW files
        if suffix in self.RAW_FORMATS:
            # Try rawpy first
            try:
                import rawpy

                with rawpy.imread(str(path)) as raw:
                    rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                    return Image.fromarray(rgb)
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"rawpy failed for {path}: {e}")

            # Try extracting embedded preview with exiftool
            try:
                import subprocess

                result = subprocess.run(
                    ["exiftool", "-b", "-PreviewImage", str(path)],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout:
                    from io import BytesIO

                    return Image.open(BytesIO(result.stdout)).convert("RGB")

                # Fallback to JpgFromRaw
                result = subprocess.run(
                    ["exiftool", "-b", "-JpgFromRaw", str(path)],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout:
                    from io import BytesIO

                    return Image.open(BytesIO(result.stdout)).convert("RGB")
            except Exception as e:
                logger.debug(f"exiftool preview extraction failed for {path}: {e}")

            return None

        # Standard image formats
        return Image.open(path).convert("RGB")

    def _load_model(self) -> None:
        """Load the CLIP model lazily."""
        if self._model is not None:
            return

        try:
            import open_clip

            model_arch, pretrained, _dim = self.MODELS.get(
                self.model_name, self.MODELS["ViT-B-32"]
            )

            logger.info(f"Loading OpenCLIP model {model_arch} ({pretrained})...")
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_arch, pretrained=pretrained, device=self._device
            )
            self._tokenizer = open_clip.get_tokenizer(model_arch)

            # Cast to float32 on CPU (bfloat16 not supported on CPU)
            if self._device == "cpu":
                self._model = self._model.float()

            self._model.eval()
            logger.info("OpenCLIP model loaded successfully")

        except ImportError as e:
            raise ImportError(
                "OpenCLIP not installed. Install with: pip install open-clip-torch"
            ) from e

    def cleanup(self) -> None:
        """Release GPU resources and clean up model.

        Call this after finishing a batch of work to free GPU memory
        for other processes.
        """
        import gc

        if self._model is not None:
            # Delete model and associated data
            del self._model
            del self._preprocess
            del self._tokenizer
            if self._text_embeddings is not None:
                del self._text_embeddings

            self._model = None
            self._preprocess = None
            self._tokenizer = None
            self._text_embeddings = None
            self._tag_names = None

            # Clear CUDA cache if using GPU
            if self._device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("GPU memory freed")
                except Exception as e:
                    logger.warning(f"Failed to clear CUDA cache: {e}")

            # Force garbage collection
            gc.collect()
            logger.info("OpenCLIP resources released")

    def _encode_tags(self, tag_names: List[str]) -> None:
        """Pre-encode tag names as text embeddings."""
        import torch

        self._load_model()

        # Check if we already have embeddings for these tags
        if self._tag_names == tag_names and self._text_embeddings is not None:
            return

        # Create text prompts for each tag
        # Using "a photo of {tag}" template improves zero-shot accuracy
        prompts = [f"a photo of {tag.replace('_', ' ')}" for tag in tag_names]

        # Use autocast only on CUDA (bfloat16 not supported on CPU)
        with torch.no_grad():
            text_tokens = self._tokenizer(prompts).to(self._device)
            if self._device == "cuda":
                with torch.amp.autocast(self._device):
                    self._text_embeddings = self._model.encode_text(text_tokens)
            else:
                self._text_embeddings = self._model.encode_text(text_tokens)
            self._text_embeddings /= self._text_embeddings.norm(dim=-1, keepdim=True)

        self._tag_names = tag_names
        logger.debug(f"Encoded {len(tag_names)} tag embeddings")

    def tag_image(
        self,
        image_path: Path,
        tag_names: List[str],
        threshold: float = 0.25,
        max_tags: int = 10,
    ) -> List[Tuple[str, float]]:
        """Tag a single image using CLIP similarity."""
        import torch

        self._load_model()
        self._encode_tags(tag_names)

        # Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return []

        # Compute image embedding and similarity
        with torch.no_grad(), torch.amp.autocast(self._device):
            image_embedding = self._model.encode_image(image_tensor)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarities = (image_embedding @ self._text_embeddings.T).squeeze(0)
            similarities = similarities.cpu().numpy()

        # Filter and sort results
        results = []
        for tag_name, score in zip(tag_names, similarities):
            if score >= threshold:
                results.append((tag_name, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_tags]

    def tag_batch(
        self,
        image_paths: List[Path],
        tag_names: List[str],
        threshold: float = 0.25,
        max_tags: int = 10,
        batch_size: int = 32,
    ) -> Dict[Path, List[Tuple[str, float]]]:
        """Tag multiple images efficiently in batches."""
        import torch

        self._load_model()
        self._encode_tags(tag_names)

        results: Dict[Path, List[Tuple[str, float]]] = {}

        # Process in batches
        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start : batch_start + batch_size]
            batch_images = []
            valid_paths = []

            # Load images (with RAW format support)
            for path in batch_paths:
                try:
                    image = self._load_image(path)
                    if image:
                        batch_images.append(self._preprocess(image))
                        valid_paths.append(path)
                    else:
                        logger.warning(f"Failed to load {path}: unsupported format")
                        results[path] = []
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    results[path] = []

            if not batch_images:
                continue

            # Stack and process batch
            batch_tensor = torch.stack(batch_images).to(self._device)

            # Use autocast only on CUDA (bfloat16 not supported on CPU)
            with torch.no_grad():
                if self._device == "cuda":
                    with torch.amp.autocast(self._device):
                        image_embeddings = self._model.encode_image(batch_tensor)
                else:
                    image_embeddings = self._model.encode_image(batch_tensor)
                image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
                similarities = image_embeddings @ self._text_embeddings.T
                similarities = similarities.cpu().numpy()

            # Extract results for each image
            for i, path in enumerate(valid_paths):
                image_results = []
                for j, tag_name in enumerate(tag_names):
                    score = float(similarities[i, j])
                    if score >= threshold:
                        image_results.append((tag_name, score))

                image_results.sort(key=lambda x: x[1], reverse=True)
                results[path] = image_results[:max_tags]

        return results

    def get_embedding(self, image_path: Union[str, Path]) -> List[float]:
        """Get CLIP embedding for an image, projected to target dimension.

        The native embedding dimension varies by model (512 for ViT-B-32,
        768 for ViT-L-14, 1024 for ViT-H-14). This method projects all
        embeddings to TARGET_EMBEDDING_DIM (768) for database storage.

        Args:
            image_path: Path to image file

        Returns:
            768-dimensional embedding as list of floats (projected if needed)
        """
        import numpy as np
        import torch

        self._load_model()

        image_path = Path(image_path)

        # Load and preprocess image
        try:
            image = self._load_image(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            raise

        # Compute image embedding
        with torch.no_grad():
            if self._device == "cuda":
                with torch.amp.autocast(self._device):
                    image_embedding = self._model.encode_image(image_tensor)
            else:
                image_embedding = self._model.encode_image(image_tensor)
            image_embedding = image_embedding / image_embedding.norm(
                dim=-1, keepdim=True
            )

        embedding = image_embedding.cpu().numpy().flatten()

        # Project to target dimension for database storage
        target_dim = self.TARGET_EMBEDDING_DIM
        if len(embedding) < target_dim:
            # Pad with zeros for smaller models (e.g., ViT-B-32: 512 -> 768)
            embedding = np.pad(embedding, (0, target_dim - len(embedding)))
        elif len(embedding) > target_dim:
            # Truncate for larger models (e.g., ViT-H-14: 1024 -> 768)
            embedding = embedding[:target_dim]

        return embedding.tolist()

    def is_available(self) -> bool:
        """Check if OpenCLIP is available."""
        try:
            import open_clip  # noqa: F401

            return True
        except ImportError:
            return False


class OllamaBackend(TaggerBackend):
    """Ollama-based vision language model backend.

    Uses LLaVA or other vision models through Ollama for more detailed
    image understanding. Slower but can provide richer descriptions.
    """

    # Supported vision models
    VISION_MODELS = ["llava", "llava:13b", "llava:34b", "qwen3-vl", "llava-llama3"]

    # Maximum dimension for images sent to Ollama (resize larger images)
    MAX_IMAGE_DIMENSION = 1024

    # RAW formats that need special handling
    RAW_FORMATS = {
        ".arw",
        ".cr2",
        ".cr3",
        ".nef",
        ".dng",
        ".orf",
        ".rw2",
        ".pef",
        ".srw",
        ".raf",
        ".raw",
    }

    # HEIC/HEIF formats
    HEIC_FORMATS = {".heic", ".heif"}

    def __init__(
        self,
        model: str = "llava",
        host: Optional[str] = None,
    ) -> None:
        """Initialize Ollama backend.

        Args:
            model: Vision model to use (llava, qwen3-vl, etc.)
            host: Ollama server URL (defaults to OLLAMA_HOST env var or localhost)
        """
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create Ollama client."""
        if self._client is None:
            try:
                import ollama

                self._client = ollama.Client(host=self.host)
            except ImportError as e:
                raise ImportError(
                    "Ollama not installed. Install with: pip install ollama"
                ) from e
        return self._client

    def _load_image(self, path: Path) -> Optional[Image.Image]:
        """Load an image, with RAW and HEIC format support.

        Args:
            path: Path to the image file

        Returns:
            PIL Image in RGB mode, or None if loading failed
        """
        from io import BytesIO

        suffix = path.suffix.lower()

        # Handle RAW files
        if suffix in self.RAW_FORMATS:
            # Try rawpy first
            try:
                import rawpy

                with rawpy.imread(str(path)) as raw:
                    rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                    return Image.fromarray(rgb)
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"rawpy failed for {path}: {e}")

            # Try extracting embedded preview with exiftool
            try:
                import subprocess

                result = subprocess.run(
                    ["exiftool", "-b", "-PreviewImage", str(path)],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout:
                    return Image.open(BytesIO(result.stdout)).convert("RGB")

                # Fallback to JpgFromRaw
                result = subprocess.run(
                    ["exiftool", "-b", "-JpgFromRaw", str(path)],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout:
                    return Image.open(BytesIO(result.stdout)).convert("RGB")
            except Exception as e:
                logger.debug(f"exiftool preview extraction failed for {path}: {e}")

            return None

        # Handle HEIC/HEIF files
        if suffix in self.HEIC_FORMATS:
            try:
                # Try pillow-heif
                import pillow_heif

                heif_file = pillow_heif.read_heif(str(path))
                return Image.frombytes(
                    heif_file.mode, heif_file.size, heif_file.data, "raw"
                ).convert("RGB")
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"pillow-heif failed for {path}: {e}")

            # Try pyheif
            try:
                import pyheif

                heif_file = pyheif.read(str(path))
                return Image.frombytes(
                    heif_file.mode, heif_file.size, heif_file.data, "raw"
                ).convert("RGB")
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"pyheif failed for {path}: {e}")

            # Try extracting preview with exiftool
            try:
                import subprocess

                result = subprocess.run(
                    ["exiftool", "-b", "-PreviewImage", str(path)],
                    capture_output=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout:
                    return Image.open(BytesIO(result.stdout)).convert("RGB")
            except Exception as e:
                logger.debug(f"exiftool preview extraction failed for {path}: {e}")

            return None

        # Standard image formats
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.debug(f"PIL failed to open {path}: {e}")
            return None

    def _prepare_image_bytes(self, image_path: Path) -> Optional[bytes]:
        """Load, resize, and convert image to JPEG bytes for Ollama.

        This handles:
        - RAW formats (.arw, .cr2, .nef, etc.)
        - HEIC/HEIF formats
        - Large images (resized to MAX_IMAGE_DIMENSION)
        - Converting to JPEG for Ollama compatibility

        Args:
            image_path: Path to the image file

        Returns:
            JPEG bytes suitable for Ollama, or None if loading failed
        """
        from io import BytesIO

        # Load image (handles RAW, HEIC, and standard formats)
        image = self._load_image(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return None

        # Resize if too large
        max_dim = max(image.size)
        if max_dim > self.MAX_IMAGE_DIMENSION:
            ratio = self.MAX_IMAGE_DIMENSION / max_dim
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(
                f"Resized {image_path.name} from {max_dim}px to {max(new_size)}px"
            )

        # Convert to JPEG bytes
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    def tag_image(
        self,
        image_path: Path,
        tag_names: List[str],
        threshold: float = 0.25,
        max_tags: int = 10,
    ) -> List[Tuple[str, float]]:
        """Tag image using Ollama vision model."""
        client = self._get_client()

        # Build prompt with available tags
        tag_list = ", ".join(tag_names)
        prompt = f"""Analyze this image and identify which of the following tags apply.
Return ONLY a JSON object with tag names as keys and confidence scores (0.0-1.0) as values.
Only include tags with confidence >= {threshold}.

Available tags: {tag_list}

Example response format:
{{"dogs": 0.95, "outdoor": 0.8, "daylight": 0.7}}

Respond with ONLY the JSON object, no other text."""

        try:
            # Load, resize, and convert image to JPEG for Ollama
            image_bytes = self._prepare_image_bytes(image_path)
            if image_bytes is None:
                logger.warning(f"Could not prepare image for Ollama: {image_path}")
                return []

            response = client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_bytes],
                    }
                ],
            )

            # Parse JSON response
            content = response["message"]["content"].strip()

            # Try to extract JSON from response
            if content.startswith("```"):
                # Remove code blocks
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            tag_scores = json.loads(content)

            # Convert to list of tuples
            results = [
                (tag, float(score))
                for tag, score in tag_scores.items()
                if tag in tag_names and float(score) >= threshold
            ]
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_tags]

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Ollama response as JSON: {e}")
            return []
        except Exception as e:
            logger.warning(f"Ollama tagging failed for {image_path}: {e}")
            return []

    def tag_batch(
        self,
        image_paths: List[Path],
        tag_names: List[str],
        threshold: float = 0.25,
        max_tags: int = 10,
    ) -> Dict[Path, List[Tuple[str, float]]]:
        """Tag multiple images (processes sequentially)."""
        results = {}
        for path in image_paths:
            results[path] = self.tag_image(path, tag_names, threshold, max_tags)
        return results

    def is_available(self) -> bool:
        """Check if Ollama is available and has vision model."""
        try:
            client = self._get_client()
            # Check if model is available
            response = client.list()
            # Handle both dict response and object response
            if hasattr(response, "models"):
                models_list = response.models
            else:
                models_list = response.get("models", [])

            model_names = []
            for m in models_list:
                # Ollama client uses 'model' attribute for model name
                if hasattr(m, "model"):
                    name = m.model
                elif hasattr(m, "name"):
                    name = m.name
                else:
                    name = m.get("model", m.get("name", ""))
                model_names.append(name.split(":")[0])

            return self.model.split(":")[0] in model_names
        except Exception:
            return False

    def describe_image(self, image_path: Path) -> str:
        """Get a natural language description of the image.

        This can be used to generate custom tags or detailed captions.
        """
        client = self._get_client()

        prompt = """Describe this image in detail. Include:
- Main subjects (people, animals, objects)
- Setting/location (indoor, outdoor, urban, nature)
- Lighting conditions
- Mood or atmosphere
- Any notable activities or interactions

Be concise but thorough."""

        try:
            # Load, resize, and convert image to JPEG for Ollama
            image_bytes = self._prepare_image_bytes(image_path)
            if image_bytes is None:
                logger.warning(f"Could not prepare image for Ollama: {image_path}")
                return ""

            response = client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_bytes],
                    }
                ],
            )
            return response["message"]["content"]
        except Exception as e:
            logger.warning(f"Failed to describe image {image_path}: {e}")
            return ""


class ImageTagger:
    """High-level image tagger combining taxonomy with AI backends.

    This class provides a unified interface for tagging images using
    the predefined tag taxonomy and either OpenCLIP or Ollama backends.

    Example:
        >>> tagger = ImageTagger(backend="openclip")
        >>> results = tagger.tag_image("/path/to/photo.jpg")
        >>> for tag in results:
        ...     print(f"{tag.tag_name}: {tag.confidence:.2f}")
        dogs: 0.85
        outdoor: 0.72
        daylight: 0.68
    """

    def __init__(
        self,
        backend: str = "openclip",
        model: Optional[str] = None,
        device: Optional[str] = None,
        ollama_host: Optional[str] = None,
    ) -> None:
        """Initialize image tagger.

        Args:
            backend: Backend to use ("openclip" or "ollama")
            model: Model name (backend-specific)
            device: Device for OpenCLIP (cuda, cpu, mps)
            ollama_host: Ollama server URL (defaults to OLLAMA_HOST env var)
        """
        self.taxonomy = TagTaxonomy()
        self._tag_names = [tag.name for tag in self.taxonomy.get_all_tags()]

        # Initialize backend
        if backend == "openclip":
            model = model or "ViT-B-32"
            self._backend: TaggerBackend = OpenCLIPBackend(
                model_name=model, device=device
            )
            self._source = "openclip"
        elif backend == "ollama":
            model = model or "llava"
            self._backend = OllamaBackend(model=model, host=ollama_host)
            self._source = "ollama"
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'openclip' or 'ollama'")

        logger.info(f"ImageTagger initialized with {backend} backend")

    def is_available(self) -> bool:
        """Check if the tagger backend is available."""
        return self._backend.is_available()

    def tag_image(
        self,
        image_path: Union[str, Path],
        threshold: float = 0.25,
        max_tags: int = 10,
        include_parents: bool = True,
    ) -> List[TagResult]:
        """Tag a single image.

        Args:
            image_path: Path to image file
            threshold: Minimum confidence threshold
            max_tags: Maximum tags to return
            include_parents: Also add parent tags with reduced confidence

        Returns:
            List of TagResult objects
        """
        image_path = Path(image_path)

        raw_results = self._backend.tag_image(
            image_path, self._tag_names, threshold, max_tags
        )

        results = []
        seen_tags = set()

        for tag_name, confidence in raw_results:
            tag_def = self.taxonomy.get_tag_by_name(tag_name)
            if tag_def:
                results.append(
                    TagResult(
                        tag_name=tag_name,
                        confidence=confidence,
                        category=tag_def.category.value,
                        source=self._source,
                    )
                )
                seen_tags.add(tag_name)

                # Add parent tags with reduced confidence
                if include_parents and tag_def.parent_id:
                    parent = self.taxonomy.get_tag_by_id(tag_def.parent_id)
                    while parent and parent.name not in seen_tags:
                        parent_conf = confidence * 0.8  # Reduce confidence for parents
                        if parent_conf >= threshold:
                            results.append(
                                TagResult(
                                    tag_name=parent.name,
                                    confidence=parent_conf,
                                    category=parent.category.value,
                                    source=self._source,
                                )
                            )
                            seen_tags.add(parent.name)
                        if parent.parent_id:
                            parent = self.taxonomy.get_tag_by_id(parent.parent_id)
                        else:
                            break

        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:max_tags]

    def tag_batch(
        self,
        image_paths: List[Union[str, Path]],
        threshold: float = 0.25,
        max_tags: int = 10,
        include_parents: bool = True,
    ) -> Dict[Path, List[TagResult]]:
        """Tag multiple images in batch.

        Args:
            image_paths: List of paths to image files
            threshold: Minimum confidence threshold
            max_tags: Maximum tags per image
            include_parents: Also add parent tags

        Returns:
            Dictionary mapping paths to tag results
        """
        paths = [Path(p) for p in image_paths]

        raw_results = self._backend.tag_batch(
            paths, self._tag_names, threshold, max_tags * 2  # Get extra for parents
        )

        final_results: Dict[Path, List[TagResult]] = {}

        for path, tags in raw_results.items():
            results = []
            seen_tags = set()

            for tag_name, confidence in tags:
                tag_def = self.taxonomy.get_tag_by_name(tag_name)
                if tag_def:
                    results.append(
                        TagResult(
                            tag_name=tag_name,
                            confidence=confidence,
                            category=tag_def.category.value,
                            source=self._source,
                        )
                    )
                    seen_tags.add(tag_name)

                    # Add parent tags
                    if include_parents and tag_def.parent_id:
                        parent = self.taxonomy.get_tag_by_id(tag_def.parent_id)
                        while parent and parent.name not in seen_tags:
                            parent_conf = confidence * 0.8
                            if parent_conf >= threshold:
                                results.append(
                                    TagResult(
                                        tag_name=parent.name,
                                        confidence=parent_conf,
                                        category=parent.category.value,
                                        source=self._source,
                                    )
                                )
                                seen_tags.add(parent.name)
                            if parent.parent_id:
                                parent = self.taxonomy.get_tag_by_id(parent.parent_id)
                            else:
                                break

            results.sort(key=lambda x: x.confidence, reverse=True)
            final_results[path] = results[:max_tags]

        return final_results

    def get_available_tags(self) -> List[str]:
        """Get list of all available tag names."""
        return self._tag_names.copy()

    def get_embedding(self, image_path: Union[str, Path]) -> List[float]:
        """Get CLIP embedding for an image, projected to 768 dimensions.

        The native embedding dimension varies by model, but embeddings are
        projected to 768 dimensions for database storage compatibility.

        Args:
            image_path: Path to image file

        Returns:
            768-dimensional embedding as list of floats (projected if needed)

        Raises:
            AttributeError: If backend doesn't support embeddings (e.g., Ollama)
        """
        if not hasattr(self._backend, "get_embedding"):
            raise AttributeError(
                f"{self._backend.__class__.__name__} backend does not support get_embedding. "
                "Only OpenCLIP backend supports CLIP embeddings."
            )

        return self._backend.get_embedding(Path(image_path))

    def describe_image(self, image_path: Union[str, Path]) -> str:
        """Get natural language description (Ollama backend only).

        Args:
            image_path: Path to image file

        Returns:
            Natural language description of the image
        """
        if not isinstance(self._backend, OllamaBackend):
            raise ValueError("describe_image only available with Ollama backend")
        return self._backend.describe_image(Path(image_path))

    def cleanup(self) -> None:
        """Release GPU resources.

        Call this after finishing processing to free GPU memory for other tasks.
        Only has an effect with OpenCLIP backend; Ollama uses external server.
        """
        if hasattr(self._backend, "cleanup"):
            self._backend.cleanup()


class CombinedTagger:
    """Combined tagger that runs both OpenCLIP and Ollama backends.

    Runs OpenCLIP first (fast batch processing), then Ollama (more accurate)
    on all images. Results are merged with weighted confidence:
    - Tags from both: 40% OpenCLIP + 60% Ollama confidence
    - Tags from one backend only: that backend's confidence

    This provides the speed of OpenCLIP with the accuracy of Ollama.
    """

    # Weights for combining confidence scores
    OPENCLIP_WEIGHT = 0.4
    OLLAMA_WEIGHT = 0.6

    def __init__(
        self,
        openclip_model: str = "ViT-B-32",
        ollama_model: str = "llava",
        device: Optional[str] = None,
        ollama_host: Optional[str] = None,
    ) -> None:
        """Initialize combined tagger with both backends.

        Args:
            openclip_model: OpenCLIP model to use
            ollama_model: Ollama vision model to use
            device: Device for OpenCLIP (cuda, cpu, mps)
            ollama_host: Ollama server URL
        """
        self.taxonomy = TagTaxonomy()
        self._tag_names = [tag.name for tag in self.taxonomy.get_all_tags()]

        # Initialize both backends
        self._openclip = OpenCLIPBackend(model_name=openclip_model, device=device)
        self._ollama = OllamaBackend(model=ollama_model, host=ollama_host)

        logger.info(
            f"CombinedTagger initialized with OpenCLIP ({openclip_model}) "
            f"and Ollama ({ollama_model})"
        )

    def is_available(self) -> Tuple[bool, bool]:
        """Check if backends are available.

        Returns:
            Tuple of (openclip_available, ollama_available)
        """
        return (self._openclip.is_available(), self._ollama.is_available())

    def _merge_results(
        self,
        openclip_tags: List[Tuple[str, float]],
        ollama_tags: List[Tuple[str, float]],
        threshold: float,
        max_tags: int,
    ) -> List[TagResult]:
        """Merge results from both backends with weighted confidence.

        Args:
            openclip_tags: Tags from OpenCLIP [(name, confidence), ...]
            ollama_tags: Tags from Ollama [(name, confidence), ...]
            threshold: Minimum confidence threshold
            max_tags: Maximum tags to return

        Returns:
            List of TagResult with combined confidence scores
        """
        # Build lookup dicts
        openclip_dict = {name: conf for name, conf in openclip_tags}
        ollama_dict = {name: conf for name, conf in ollama_tags}

        # Get all unique tag names
        all_tags = set(openclip_dict.keys()) | set(ollama_dict.keys())

        results = []
        for tag_name in all_tags:
            openclip_conf = openclip_dict.get(tag_name)
            ollama_conf = ollama_dict.get(tag_name)

            # Calculate combined confidence
            if openclip_conf is not None and ollama_conf is not None:
                # Both backends found this tag - weighted combination
                combined_conf = (
                    self.OPENCLIP_WEIGHT * openclip_conf
                    + self.OLLAMA_WEIGHT * ollama_conf
                )
                source = "combined"
            elif openclip_conf is not None:
                # Only OpenCLIP found this tag
                combined_conf = openclip_conf
                source = "openclip"
            else:
                # Only Ollama found this tag
                combined_conf = ollama_conf if ollama_conf is not None else 0.0
                source = "ollama"

            if combined_conf >= threshold:
                # Look up category from taxonomy
                tag_def = self.taxonomy.get_tag_by_name(tag_name)
                category = tag_def.category.value if tag_def else "unknown"

                results.append(
                    TagResult(
                        tag_name=tag_name,
                        confidence=combined_conf,
                        category=category,
                        source=source,
                        openclip_confidence=openclip_conf,
                        ollama_confidence=ollama_conf,
                    )
                )

        # Sort by confidence and limit
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:max_tags]

    def tag_image(
        self,
        image_path: Union[str, Path],
        threshold: float = 0.25,
        max_tags: int = 10,
    ) -> List[TagResult]:
        """Tag a single image using both backends.

        Args:
            image_path: Path to the image
            threshold: Minimum confidence threshold
            max_tags: Maximum tags to return

        Returns:
            List of TagResult with combined results
        """
        image_path = Path(image_path)

        # Get tags from OpenCLIP
        try:
            openclip_tags = self._openclip.tag_image(
                image_path, self._tag_names, threshold=0.1, max_tags=max_tags * 2
            )
        except Exception as e:
            logger.warning(f"OpenCLIP tagging failed: {e}")
            openclip_tags = []

        # Get tags from Ollama
        try:
            ollama_tags = self._ollama.tag_image(
                image_path, self._tag_names, threshold=0.1, max_tags=max_tags * 2
            )
        except Exception as e:
            logger.warning(f"Ollama tagging failed: {e}")
            ollama_tags = []

        # Merge results
        return self._merge_results(openclip_tags, ollama_tags, threshold, max_tags)

    def tag_batch(
        self,
        image_paths: List[Union[str, Path]],
        threshold: float = 0.25,
        max_tags: int = 10,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[Path, List[TagResult]]:
        """Tag multiple images using both backends.

        OpenCLIP processes the batch first (fast), then Ollama processes
        each image (slower but more accurate).

        Args:
            image_paths: List of image paths
            threshold: Minimum confidence threshold
            max_tags: Maximum tags per image
            progress_callback: Optional callback(current, total, phase) for progress

        Returns:
            Dict mapping image paths to their combined tag results
        """
        paths = [Path(p) for p in image_paths]
        total = len(paths)

        if progress_callback:
            progress_callback(0, total, "openclip")

        # Phase 1: OpenCLIP batch processing (fast)
        try:
            openclip_results = self._openclip.tag_batch(
                paths, self._tag_names, threshold=0.1, max_tags=max_tags * 2
            )
        except Exception as e:
            logger.warning(f"OpenCLIP batch tagging failed: {e}")
            openclip_results = {p: [] for p in paths}

        if progress_callback:
            progress_callback(total, total, "openclip")
            progress_callback(0, total, "ollama")

        # Phase 2: Ollama processing (slower, one at a time)
        ollama_results: Dict[Path, List[Tuple[str, float]]] = {}
        for i, path in enumerate(paths):
            try:
                ollama_results[path] = self._ollama.tag_image(
                    path, self._tag_names, threshold=0.1, max_tags=max_tags * 2
                )
            except Exception as e:
                logger.warning(f"Ollama tagging failed for {path}: {e}")
                ollama_results[path] = []

            if progress_callback:
                progress_callback(i + 1, total, "ollama")

        # Phase 3: Merge results
        combined_results: Dict[Path, List[TagResult]] = {}
        for path in paths:
            openclip_tags = openclip_results.get(path, [])
            ollama_tags = ollama_results.get(path, [])
            combined_results[path] = self._merge_results(
                openclip_tags, ollama_tags, threshold, max_tags
            )

        return combined_results

    def get_embedding(self, image_path: Union[str, Path]) -> List[float]:
        """Get CLIP embedding for an image, projected to 768 dimensions.

        Uses the OpenCLIP backend to compute the embedding. The embedding is
        projected to 768 dimensions for database storage compatibility.

        Args:
            image_path: Path to image file

        Returns:
            768-dimensional embedding as list of floats (projected if needed)
        """
        return self._openclip.get_embedding(Path(image_path))

    def cleanup(self) -> None:
        """Release GPU resources from OpenCLIP backend.

        Call this after finishing processing to free GPU memory for other tasks.
        Ollama backend uses external server, so no cleanup needed for it.
        """
        self._openclip.cleanup()


def check_backends_available() -> Dict[str, Any]:
    """Check which tagging backends are available.

    Returns:
        Dictionary with backend names and availability status
    """
    import importlib.util

    status: Dict[str, Any] = {}

    # Check OpenCLIP - use importlib to check if package exists without importing
    # This avoids triggering PyTorch CUDA initialization which can fail on
    # unsupported GPU architectures (e.g., RTX 5060 Ti Blackwell sm_120)
    open_clip_spec = importlib.util.find_spec("open_clip")
    if open_clip_spec is not None:
        status["openclip"] = True
        status["openclip_note"] = "available (will use CPU if GPU unsupported)"
    else:
        status["openclip"] = False
        status["openclip_error"] = "open_clip not installed"

    # Check Ollama
    try:
        import ollama

        client = ollama.Client()
        response = client.list()
        # Handle both dict response and object response
        if hasattr(response, "models"):
            models_list = response.models
        else:
            models_list = response.get("models", [])

        vision_models: List[str] = []
        for m in models_list:
            # Handle both dict and object model entries
            # Ollama client uses 'model' attribute for model name
            if hasattr(m, "model"):
                name = m.model
            elif hasattr(m, "name"):
                name = m.name
            else:
                name = m.get("model", m.get("name", ""))
            if name and any(
                v in name.lower() for v in ["llava", "qwen-vl", "qwen3-vl", "bakllava"]
            ):
                vision_models.append(name)
        status["ollama"] = len(vision_models) > 0
        status["ollama_models"] = vision_models
    except Exception as e:
        status["ollama"] = False
        status["ollama_models"] = []
        status["ollama_error"] = str(e)

    return status
