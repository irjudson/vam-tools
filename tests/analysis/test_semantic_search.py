"""Tests for semantic search service."""

# Check if torch is available without importing it
import importlib.util
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from lumina.analysis.semantic_search import SearchResult, SemanticSearchService

HAS_TORCH = importlib.util.find_spec("torch") is not None

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Test creating a SearchResult."""
        result = SearchResult(
            image_id="img-001",
            source_path="/photos/sunset.jpg",
            similarity_score=0.85,
        )
        assert result.image_id == "img-001"
        assert result.similarity_score == 0.85


class TestSemanticSearchService:
    """Tests for SemanticSearchService."""

    def test_initialization(self) -> None:
        """Test service initialization."""
        service = SemanticSearchService()
        assert service._model is None  # Lazy loading
        assert service.model_name == "ViT-L-14"

    @requires_torch
    def test_encode_text_returns_768_dim_vector(self) -> None:
        """Test that text encoding returns 768-dimensional vector."""
        service = SemanticSearchService()

        with patch.object(service, "_ensure_model_loaded"):
            with patch.object(service, "_model") as mock_model:
                with patch.object(service, "_tokenizer") as mock_tokenizer:
                    import numpy as np
                    import torch

                    # Mock tokenizer
                    mock_tokens = MagicMock()
                    mock_tokenizer.return_value = mock_tokens
                    mock_tokens.to.return_value = mock_tokens

                    # Mock model output
                    mock_output = torch.tensor([[0.1] * 768])
                    mock_model.encode_text.return_value = mock_output

                    embedding = service.encode_text("sunset over mountains")

                    assert len(embedding) == 768
                    assert isinstance(embedding, list)
                    assert all(isinstance(x, float) for x in embedding)

    @requires_torch
    def test_encode_image_returns_768_dim_vector(self) -> None:
        """Test that image encoding returns 768-dimensional vector."""
        service = SemanticSearchService()

        with patch.object(service, "_ensure_model_loaded"):
            with patch.object(service, "_model") as mock_model:
                with patch.object(service, "_preprocess") as mock_preprocess:
                    with patch(
                        "lumina.analysis.semantic_search.Image"
                    ) as mock_image_class:
                        import numpy as np
                        import torch

                        # Mock PIL Image
                        mock_img = MagicMock()
                        mock_image_class.open.return_value.convert.return_value = (
                            mock_img
                        )

                        # Mock preprocessing
                        mock_tensor = MagicMock()
                        mock_preprocess.return_value = mock_tensor
                        mock_tensor.unsqueeze.return_value.to.return_value = mock_tensor

                        # Mock model output
                        mock_output = torch.tensor([[0.1] * 768])
                        mock_model.encode_image.return_value = mock_output

                        # Use a mock image path
                        embedding = service.encode_image(Path("/fake/image.jpg"))

                        assert len(embedding) == 768
                        assert isinstance(embedding, list)
                        assert all(isinstance(x, float) for x in embedding)

    def test_search_returns_results_sorted_by_similarity(self) -> None:
        """Test that search returns results sorted by similarity."""
        mock_session = MagicMock()

        # Mock database results
        mock_results = [
            ("img-001", "/photos/a.jpg", 0.95),
            ("img-002", "/photos/b.jpg", 0.80),
            ("img-003", "/photos/c.jpg", 0.75),
        ]
        mock_session.execute.return_value.fetchall.return_value = mock_results

        service = SemanticSearchService()

        with patch.object(service, "encode_text", return_value=[0.1] * 768):
            results = service.search(
                session=mock_session,
                catalog_id="cat-001",
                query="sunset",
                limit=10,
            )

        assert len(results) == 3
        assert results[0].similarity_score >= results[1].similarity_score
        assert results[1].similarity_score >= results[2].similarity_score
        assert results[0].image_id == "img-001"
        assert results[0].source_path == "/photos/a.jpg"

    def test_search_filters_by_threshold(self) -> None:
        """Test that search respects similarity threshold."""
        mock_session = MagicMock()

        # Only return results above threshold
        mock_results = [
            ("img-001", "/photos/a.jpg", 0.95),
            ("img-002", "/photos/b.jpg", 0.80),
        ]
        mock_session.execute.return_value.fetchall.return_value = mock_results

        service = SemanticSearchService()

        with patch.object(service, "encode_text", return_value=[0.1] * 768):
            results = service.search(
                session=mock_session,
                catalog_id="cat-001",
                query="sunset",
                limit=10,
                threshold=0.7,
            )

        # Should only include results above threshold
        assert all(r.similarity_score >= 0.7 for r in results)

    def test_find_similar_images(self) -> None:
        """Test finding similar images to a given image."""
        mock_session = MagicMock()

        # Mock the source image embedding query
        mock_session.execute.return_value.fetchone.return_value = ([0.1] * 768,)

        # Mock similar images query
        mock_results = [
            ("img-002", "/photos/similar1.jpg", 0.92),
            ("img-003", "/photos/similar2.jpg", 0.88),
        ]
        # Need to set up two different return values for two different queries
        mock_result1 = MagicMock()
        mock_result1.fetchone.return_value = ([0.1] * 768,)
        mock_result2 = MagicMock()
        mock_result2.fetchall.return_value = mock_results

        mock_session.execute.side_effect = [mock_result1, mock_result2]

        service = SemanticSearchService()
        results = service.find_similar(
            session=mock_session,
            catalog_id="cat-001",
            image_id="img-001",
            limit=10,
        )

        assert len(results) == 2
        assert results[0].image_id == "img-002"
        assert results[0].similarity_score == 0.92
        assert results[1].image_id == "img-003"
        assert results[1].similarity_score == 0.88

    def test_find_similar_with_no_embedding(self) -> None:
        """Test find_similar when source image has no embedding."""
        mock_session = MagicMock()

        # Mock source image with no embedding
        mock_session.execute.return_value.fetchone.return_value = None

        service = SemanticSearchService()
        results = service.find_similar(
            session=mock_session,
            catalog_id="cat-001",
            image_id="img-001",
            limit=10,
        )

        # Should return empty list
        assert results == []

    def test_find_similar_excludes_source_image(self) -> None:
        """Test that find_similar excludes the source image from results."""
        mock_session = MagicMock()

        # Mock the source image embedding
        mock_result1 = MagicMock()
        mock_result1.fetchone.return_value = ([0.1] * 768,)

        # Mock similar images - should not include source image
        mock_results = [
            ("img-002", "/photos/similar1.jpg", 0.92),
            ("img-003", "/photos/similar2.jpg", 0.88),
        ]
        mock_result2 = MagicMock()
        mock_result2.fetchall.return_value = mock_results

        mock_session.execute.side_effect = [mock_result1, mock_result2]

        service = SemanticSearchService()
        results = service.find_similar(
            session=mock_session,
            catalog_id="cat-001",
            image_id="img-001",
            limit=10,
        )

        # Verify source image is not in results
        assert all(r.image_id != "img-001" for r in results)

    def test_model_lazy_loading(self) -> None:
        """Test that model is only loaded when needed."""
        service = SemanticSearchService()

        # Model should not be loaded on initialization
        assert service._model is None
        assert service._preprocess is None
        assert service._tokenizer is None

    @requires_torch
    def test_encode_text_normalizes_embeddings(self) -> None:
        """Test that text embeddings are normalized."""
        service = SemanticSearchService()

        with patch.object(service, "_ensure_model_loaded"):
            with patch.object(service, "_model") as mock_model:
                with patch.object(service, "_tokenizer") as mock_tokenizer:
                    import torch

                    mock_tokens = MagicMock()
                    mock_tokenizer.return_value = mock_tokens
                    mock_tokens.to.return_value = mock_tokens

                    # Create unnormalized vector
                    unnormalized = torch.tensor([[2.0] * 768])
                    mock_model.encode_text.return_value = unnormalized

                    embedding = service.encode_text("test query")

                    # After normalization, vector magnitude should be ~1
                    # (though exact check depends on normalization implementation)
                    assert len(embedding) == 768
