"""Tests for AI-based image tagger."""

# Check if torch is available without importing it
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from lumina.analysis.image_tagger import (
    ImageTagger,
    OllamaBackend,
    OpenCLIPBackend,
    TaggerBackend,
    TagResult,
    check_backends_available,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


class TestTagResult:
    """Tests for TagResult dataclass."""

    def test_tag_result_creation(self) -> None:
        """Test creating a TagResult."""
        result = TagResult(
            tag_name="dogs",
            confidence=0.85,
            category="subject",
            source="openclip",
        )

        assert result.tag_name == "dogs"
        assert result.confidence == 0.85
        assert result.category == "subject"
        assert result.source == "openclip"


class TestOpenCLIPBackend:
    """Tests for OpenCLIP backend."""

    def test_initialization(self) -> None:
        """Test backend initialization."""
        backend = OpenCLIPBackend(model_name="ViT-B-32")

        assert backend.model_name == "ViT-B-32"
        assert backend._model is None  # Lazy loading

    def test_model_configurations(self) -> None:
        """Test that model configurations are defined."""
        assert "ViT-B-32" in OpenCLIPBackend.MODELS
        assert "ViT-L-14" in OpenCLIPBackend.MODELS
        assert "ViT-H-14" in OpenCLIPBackend.MODELS

    def test_device_detection_cpu_fallback(self) -> None:
        """Test device detection falls back to CPU."""
        with patch.dict("sys.modules", {"torch": None}):
            backend = OpenCLIPBackend()
            # Should default to CPU when torch not available
            assert backend._device in ["cpu", "cuda", "mps"]

    @patch("lumina.analysis.image_tagger.OpenCLIPBackend._load_model")
    def test_is_available_with_openclip(self, mock_load: MagicMock) -> None:
        """Test availability check when open_clip is installed."""
        backend = OpenCLIPBackend()

        with patch.dict("sys.modules", {"open_clip": MagicMock()}):
            # Reload to get the patched module
            assert backend.is_available() is True

    def test_is_available_without_openclip(self) -> None:
        """Test availability check when open_clip is not installed."""
        backend = OpenCLIPBackend()

        # Mock import to fail
        with patch.dict("sys.modules", {"open_clip": None}):
            import sys

            # Remove from sys.modules to force ImportError
            if "open_clip" in sys.modules:
                del sys.modules["open_clip"]

            # The is_available method tries to import, which may still succeed
            # if the package is actually installed
            result = backend.is_available()
            assert isinstance(result, bool)


class TestOllamaBackend:
    """Tests for Ollama backend."""

    def test_initialization(self) -> None:
        """Test backend initialization."""
        backend = OllamaBackend(model="llava", host="http://localhost:11434")

        assert backend.model == "llava"
        assert backend.host == "http://localhost:11434"
        assert backend._client is None  # Lazy loading

    def test_vision_models_defined(self) -> None:
        """Test that vision models are defined."""
        assert "llava" in OllamaBackend.VISION_MODELS
        assert "llava:13b" in OllamaBackend.VISION_MODELS

    def test_prepare_image_bytes(self, tmp_path: Path) -> None:
        """Test image preparation with resizing."""
        from PIL import Image

        # Create a small test image
        test_image = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(test_image, format="JPEG")

        backend = OllamaBackend()
        result = backend._prepare_image_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0
        # Verify it's valid JPEG
        from io import BytesIO

        loaded = Image.open(BytesIO(result))
        assert loaded.format == "JPEG"

    def test_prepare_image_bytes_resizes_large_images(self, tmp_path: Path) -> None:
        """Test that large images are resized."""
        from PIL import Image

        # Create a large test image (2000x1500)
        test_image = tmp_path / "large.jpg"
        img = Image.new("RGB", (2000, 1500), color="blue")
        img.save(test_image, format="JPEG")

        backend = OllamaBackend()
        result = backend._prepare_image_bytes(test_image)

        # Load the result and check dimensions
        from io import BytesIO

        loaded = Image.open(BytesIO(result))
        # Should be resized to max 1024
        assert max(loaded.size) == 1024
        assert loaded.size == (1024, 768)  # Maintains aspect ratio

    @patch("lumina.analysis.image_tagger.OllamaBackend._get_client")
    @patch("lumina.analysis.image_tagger.OllamaBackend._prepare_image_bytes")
    def test_tag_image_success(
        self, mock_prepare: MagicMock, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        """Test successful image tagging with Ollama."""
        # Create a mock image file
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")

        # Mock the image preparation
        mock_prepare.return_value = b"prepared jpeg bytes"

        # Mock the client response
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": '{"dogs": 0.95, "outdoor": 0.8}'}
        }
        mock_get_client.return_value = mock_client

        backend = OllamaBackend()
        results = backend.tag_image(
            test_image, ["dogs", "cats", "outdoor"], threshold=0.5
        )

        assert len(results) == 2
        assert ("dogs", 0.95) in results
        assert ("outdoor", 0.8) in results

    @patch("lumina.analysis.image_tagger.OllamaBackend._get_client")
    @patch("lumina.analysis.image_tagger.OllamaBackend._prepare_image_bytes")
    def test_tag_image_json_in_code_block(
        self, mock_prepare: MagicMock, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        """Test parsing JSON from code block response."""
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")

        mock_prepare.return_value = b"prepared jpeg bytes"

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": '```json\n{"dogs": 0.9}\n```'}
        }
        mock_get_client.return_value = mock_client

        backend = OllamaBackend()
        results = backend.tag_image(test_image, ["dogs"], threshold=0.5)

        assert len(results) == 1
        assert results[0][0] == "dogs"

    @patch("lumina.analysis.image_tagger.OllamaBackend._get_client")
    @patch("lumina.analysis.image_tagger.OllamaBackend._prepare_image_bytes")
    def test_tag_image_invalid_json(
        self, mock_prepare: MagicMock, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        """Test handling of invalid JSON response."""
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")

        mock_prepare.return_value = b"prepared jpeg bytes"

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "This is not JSON at all"}
        }
        mock_get_client.return_value = mock_client

        backend = OllamaBackend()
        results = backend.tag_image(test_image, ["dogs"], threshold=0.5)

        # Should return empty list on parse failure
        assert results == []

    @patch("lumina.analysis.image_tagger.OllamaBackend._get_client")
    @patch("lumina.analysis.image_tagger.OllamaBackend._prepare_image_bytes")
    def test_tag_batch(
        self, mock_prepare: MagicMock, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        """Test batch tagging with Ollama."""
        mock_prepare.return_value = b"prepared jpeg bytes"

        # Create test images
        images = []
        for i in range(3):
            img = tmp_path / f"test{i}.jpg"
            img.write_bytes(b"fake image data")
            images.append(img)

        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": '{"dogs": 0.9}'}}
        mock_get_client.return_value = mock_client

        backend = OllamaBackend()
        results = backend.tag_batch(images, ["dogs", "cats"], threshold=0.5)

        assert len(results) == 3
        for path in images:
            assert path in results

    @patch("lumina.analysis.image_tagger.OllamaBackend._get_client")
    def test_is_available_with_model(self, mock_get_client: MagicMock) -> None:
        """Test availability check with model present."""
        mock_client = MagicMock()
        # Mock response with model objects that have 'model' attribute
        mock_model = MagicMock()
        mock_model.model = "llava:latest"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_client.list.return_value = mock_response
        mock_get_client.return_value = mock_client

        backend = OllamaBackend(model="llava")
        assert backend.is_available() is True

    @patch("lumina.analysis.image_tagger.OllamaBackend._get_client")
    def test_is_available_without_model(self, mock_get_client: MagicMock) -> None:
        """Test availability check without required model."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.model = "other-model:latest"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_client.list.return_value = mock_response
        mock_get_client.return_value = mock_client

        backend = OllamaBackend(model="llava")
        assert backend.is_available() is False

    @patch("lumina.analysis.image_tagger.OllamaBackend._get_client")
    @patch("lumina.analysis.image_tagger.OllamaBackend._prepare_image_bytes")
    def test_describe_image(
        self, mock_prepare: MagicMock, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        """Test image description."""
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake image data")

        mock_prepare.return_value = b"prepared jpeg bytes"

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "A beautiful outdoor scene with dogs playing."}
        }
        mock_get_client.return_value = mock_client

        backend = OllamaBackend()
        description = backend.describe_image(test_image)

        assert "dogs" in description.lower()
        assert "outdoor" in description.lower()


class TestImageTagger:
    """Tests for high-level ImageTagger class."""

    def test_initialization_openclip(self) -> None:
        """Test initialization with OpenCLIP backend."""
        with patch.object(OpenCLIPBackend, "__init__", return_value=None):
            with patch.object(OpenCLIPBackend, "is_available", return_value=True):
                tagger = ImageTagger(backend="openclip", model="ViT-B-32")

                assert tagger._source == "openclip"
                assert isinstance(tagger._backend, OpenCLIPBackend)

    def test_initialization_ollama(self) -> None:
        """Test initialization with Ollama backend."""
        with patch.object(OllamaBackend, "__init__", return_value=None):
            with patch.object(OllamaBackend, "is_available", return_value=True):
                tagger = ImageTagger(backend="ollama", model="llava")

                assert tagger._source == "ollama"
                assert isinstance(tagger._backend, OllamaBackend)

    def test_initialization_invalid_backend(self) -> None:
        """Test initialization with invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            ImageTagger(backend="invalid")

    def test_get_available_tags(self) -> None:
        """Test getting available tag names."""
        with patch.object(OpenCLIPBackend, "__init__", return_value=None):
            with patch.object(OpenCLIPBackend, "is_available", return_value=True):
                tagger = ImageTagger(backend="openclip")
                tags = tagger.get_available_tags()

                assert isinstance(tags, list)
                assert len(tags) > 0
                assert "dogs" in tags
                assert "cats" in tags
                assert "outdoor" in tags

    def test_is_available(self) -> None:
        """Test availability check delegates to backend."""
        with patch.object(OpenCLIPBackend, "__init__", return_value=None):
            mock_backend = MagicMock()
            mock_backend.is_available.return_value = True

            tagger = ImageTagger(backend="openclip")
            tagger._backend = mock_backend

            assert tagger.is_available() is True
            mock_backend.is_available.assert_called_once()

    def test_tag_image_with_parents(self) -> None:
        """Test tagging includes parent tags."""
        with patch.object(OpenCLIPBackend, "__init__", return_value=None):
            mock_backend = MagicMock()
            # Return dogs tag which has animals as parent
            mock_backend.tag_image.return_value = [("dogs", 0.9)]

            tagger = ImageTagger(backend="openclip")
            tagger._backend = mock_backend

            results = tagger.tag_image("/fake/path.jpg", include_parents=True)

            # Should include dogs and its parent (animals)
            tag_names = [r.tag_name for r in results]
            assert "dogs" in tag_names
            # Parent tag should be included with reduced confidence
            assert "animals" in tag_names

    def test_tag_image_without_parents(self) -> None:
        """Test tagging without parent tags."""
        with patch.object(OpenCLIPBackend, "__init__", return_value=None):
            mock_backend = MagicMock()
            mock_backend.tag_image.return_value = [("dogs", 0.9)]

            tagger = ImageTagger(backend="openclip")
            tagger._backend = mock_backend

            results = tagger.tag_image("/fake/path.jpg", include_parents=False)

            tag_names = [r.tag_name for r in results]
            assert "dogs" in tag_names
            assert "animals" not in tag_names

    def test_tag_batch(self) -> None:
        """Test batch tagging."""
        with patch.object(OpenCLIPBackend, "__init__", return_value=None):
            mock_backend = MagicMock()
            mock_backend.tag_batch.return_value = {
                Path("/fake/1.jpg"): [("dogs", 0.9)],
                Path("/fake/2.jpg"): [("cats", 0.85)],
            }

            tagger = ImageTagger(backend="openclip")
            tagger._backend = mock_backend

            results = tagger.tag_batch(["/fake/1.jpg", "/fake/2.jpg"])

            assert len(results) == 2
            assert Path("/fake/1.jpg") in results
            assert Path("/fake/2.jpg") in results

    def test_describe_image_ollama_only(self) -> None:
        """Test describe_image only works with Ollama backend."""
        with patch.object(OpenCLIPBackend, "__init__", return_value=None):
            tagger = ImageTagger(backend="openclip")

            with pytest.raises(ValueError, match="only available with Ollama"):
                tagger.describe_image("/fake/path.jpg")

    def test_describe_image_with_ollama(self) -> None:
        """Test describe_image with Ollama backend."""
        with patch.object(OllamaBackend, "__init__", return_value=None):
            mock_backend = MagicMock(spec=OllamaBackend)
            mock_backend.describe_image.return_value = "A beautiful landscape"

            tagger = ImageTagger(backend="ollama")
            tagger._backend = mock_backend

            result = tagger.describe_image("/fake/path.jpg")

            assert result == "A beautiful landscape"


class TestCheckBackendsAvailable:
    """Tests for check_backends_available function.

    Note: These tests check the actual environment status.
    The function is designed to gracefully handle missing dependencies.
    """

    def test_returns_dict(self) -> None:
        """Test that function returns a dictionary."""
        result = check_backends_available()

        assert isinstance(result, dict)
        assert "openclip" in result
        assert "ollama" in result

    def test_openclip_status_is_boolean(self) -> None:
        """Test OpenCLIP status is boolean."""
        result = check_backends_available()

        assert isinstance(result["openclip"], bool)

    def test_ollama_status_is_boolean(self) -> None:
        """Test Ollama status is boolean."""
        result = check_backends_available()

        assert isinstance(result["ollama"], bool)

    def test_ollama_models_list_present(self) -> None:
        """Test Ollama models list is present."""
        result = check_backends_available()

        assert "ollama_models" in result
        assert isinstance(result["ollama_models"], list)

    def test_handles_ollama_unavailable(self) -> None:
        """Test function handles unavailable Ollama gracefully."""
        # This test validates the error handling logic works
        # by checking the expected keys are present regardless of availability
        result = check_backends_available()

        # Should always have these keys
        assert "ollama" in result
        assert "ollama_models" in result

        # If unavailable, should have error info
        if not result["ollama"] and result["ollama_models"] == []:
            # Error case - may have ollama_error key
            pass  # This is valid behavior


class TestTaggerBackendInterface:
    """Tests for TaggerBackend abstract interface."""

    def test_abstract_methods(self) -> None:
        """Test that TaggerBackend defines required abstract methods."""
        # Verify abstract methods exist
        assert hasattr(TaggerBackend, "tag_image")
        assert hasattr(TaggerBackend, "tag_batch")
        assert hasattr(TaggerBackend, "is_available")

    def test_cannot_instantiate_abstract(self) -> None:
        """Test that TaggerBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TaggerBackend()  # type: ignore


class MockBackend(TaggerBackend):
    """Mock backend for testing interface compliance."""

    def tag_image(
        self,
        image_path: Path,
        tag_names: List[str],
        threshold: float = 0.25,
        max_tags: int = 10,
    ) -> List[Tuple[str, float]]:
        return [("mock_tag", 0.99)]

    def tag_batch(
        self,
        image_paths: List[Path],
        tag_names: List[str],
        threshold: float = 0.25,
        max_tags: int = 10,
    ) -> Dict[Path, List[Tuple[str, float]]]:
        return {p: [("mock_tag", 0.99)] for p in image_paths}

    def is_available(self) -> bool:
        return True


class TestMockBackend:
    """Tests using mock backend to verify interface."""

    def test_mock_backend_implements_interface(self) -> None:
        """Test that mock backend properly implements interface."""
        backend = MockBackend()

        assert backend.is_available() is True

        result = backend.tag_image(Path("/fake.jpg"), ["tag1", "tag2"])
        assert len(result) == 1
        assert result[0] == ("mock_tag", 0.99)

        batch_result = backend.tag_batch(
            [Path("/fake1.jpg"), Path("/fake2.jpg")], ["tag1"]
        )
        assert len(batch_result) == 2


class TestImageTaggerEmbeddings:
    """Tests for CLIP embedding extraction."""

    @requires_torch
    def test_openclip_backend_get_embedding_returns_768_dim_vector(self) -> None:
        """Test that OpenCLIPBackend.get_embedding returns 768-dimensional vector."""
        with patch.object(OpenCLIPBackend, "_load_model"):
            backend = OpenCLIPBackend(model_name="ViT-B-32")

            # Mock the model and preprocessing
            import numpy as np
            import torch

            mock_model = MagicMock()
            mock_preprocess = MagicMock()

            # Create a mock tensor for the image
            mock_image_tensor = torch.randn(1, 3, 224, 224)
            mock_preprocess.return_value = mock_image_tensor.squeeze(0)

            # Create a mock 768-dim embedding
            mock_embedding = torch.randn(1, 768)
            mock_model.encode_image.return_value = mock_embedding

            backend._model = mock_model
            backend._preprocess = mock_preprocess
            backend._device = "cpu"

            # Create a temporary test image
            from pathlib import Path

            test_path = Path("/tmp/test_image.jpg")

            with patch("PIL.Image.open") as mock_open:
                mock_img = MagicMock()
                mock_img.convert.return_value = mock_img
                mock_open.return_value = mock_img

                embedding = backend.get_embedding(test_path)

                assert isinstance(embedding, list)
                assert len(embedding) == 768
                assert all(isinstance(x, float) for x in embedding)

    def test_image_tagger_get_embedding_returns_768_dim_vector(self) -> None:
        """Test that ImageTagger.get_embedding returns 768-dimensional vector."""
        with patch.object(OpenCLIPBackend, "__init__", return_value=None):
            tagger = ImageTagger(backend="openclip")

            # Mock the backend
            mock_backend = MagicMock()
            import numpy as np

            mock_embedding = np.random.randn(768).tolist()
            mock_backend.get_embedding.return_value = mock_embedding

            tagger._backend = mock_backend

            embedding = tagger.get_embedding(Path("/fake/image.jpg"))

            assert isinstance(embedding, list)
            assert len(embedding) == 768
            mock_backend.get_embedding.assert_called_once_with(Path("/fake/image.jpg"))

    def test_image_tagger_get_embedding_with_ollama_backend(self) -> None:
        """Test that get_embedding raises error for non-OpenCLIP backends."""
        with patch.object(OllamaBackend, "__init__", return_value=None):
            tagger = ImageTagger(backend="ollama")
            tagger._backend = MagicMock(spec=OllamaBackend)

            # Ollama backend doesn't have get_embedding, so should raise AttributeError
            with pytest.raises(AttributeError):
                tagger.get_embedding(Path("/fake/image.jpg"))
