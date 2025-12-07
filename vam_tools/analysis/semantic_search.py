"""Semantic search service using CLIP embeddings.

Enables natural language image search by encoding text queries
and finding visually similar images via vector similarity.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

from PIL import Image
from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    image_id: str
    source_path: str
    similarity_score: float
    thumbnail_path: Optional[str] = None


class SemanticSearchService:
    """Service for semantic image search using CLIP embeddings.

    Uses the same OpenCLIP model as the image tagger, so embeddings
    are computed during tagging with zero additional overhead.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
    ):
        """Initialize the semantic search service.

        Args:
            model_name: OpenCLIP model name
            pretrained: Pretrained weights identifier
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._device = "cpu"

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the CLIP model."""
        if self._model is not None:
            return

        try:
            import open_clip
            import torch

            # Detect device
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

            logger.info(f"Loading OpenCLIP {self.model_name} on {self._device}")

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            self._model = self._model.to(self._device)
            self._model.eval()

            self._tokenizer = open_clip.get_tokenizer(self.model_name)

            logger.info("OpenCLIP model loaded successfully")

        except ImportError:
            raise ImportError(
                "open_clip is required for semantic search. "
                "Install with: pip install open-clip-torch"
            )

    def encode_text(self, query: str) -> List[float]:
        """Encode a text query to a CLIP embedding.

        Args:
            query: Text query (e.g., "sunset over mountains")

        Returns:
            768-dimensional embedding as list of floats
        """
        self._ensure_model_loaded()

        import torch

        with torch.no_grad():
            tokens = self._tokenizer([query]).to(self._device)
            text_features = self._model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().flatten().tolist()

    def encode_image(self, image_path: Union[str, Path]) -> List[float]:
        """Encode an image to a CLIP embedding.

        Args:
            image_path: Path to image file

        Returns:
            768-dimensional embedding as list of floats
        """
        self._ensure_model_loaded()

        import torch

        image = Image.open(image_path).convert("RGB")
        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().flatten().tolist()

    def search(
        self,
        session: Any,
        catalog_id: str,
        query: str,
        limit: int = 50,
        threshold: float = 0.2,
    ) -> List[SearchResult]:
        """Search for images matching a text query.

        Args:
            session: SQLAlchemy session
            catalog_id: Catalog ID to search in
            query: Text query
            limit: Maximum results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of SearchResults sorted by similarity (highest first)
        """
        query_embedding = self.encode_text(query)

        # Use pgvector cosine similarity
        result = session.execute(
            text(
                """
                SELECT
                    i.id,
                    i.source_path,
                    1 - (i.clip_embedding <=> :embedding::vector) as similarity
                FROM images i
                WHERE i.catalog_id = :catalog_id
                AND i.clip_embedding IS NOT NULL
                AND 1 - (i.clip_embedding <=> :embedding::vector) >= :threshold
                ORDER BY i.clip_embedding <=> :embedding::vector
                LIMIT :limit
            """
            ),
            {
                "catalog_id": catalog_id,
                "embedding": str(query_embedding),
                "threshold": threshold,
                "limit": limit,
            },
        )

        return [
            SearchResult(
                image_id=row[0],
                source_path=row[1],
                similarity_score=float(row[2]),
            )
            for row in result.fetchall()
        ]

    def find_similar(
        self,
        session: Any,
        catalog_id: str,
        image_id: str,
        limit: int = 20,
        threshold: float = 0.5,
    ) -> List[SearchResult]:
        """Find images similar to a given image.

        Args:
            session: SQLAlchemy session
            catalog_id: Catalog ID to search in
            image_id: Source image ID
            limit: Maximum results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of similar images (excludes the source image)
        """
        # Get the source image's embedding
        result = session.execute(
            text(
                """
                SELECT clip_embedding
                FROM images
                WHERE id = :image_id AND catalog_id = :catalog_id
            """
            ),
            {"image_id": image_id, "catalog_id": catalog_id},
        )
        row = result.fetchone()

        if row is None or row[0] is None:
            logger.warning(f"Image {image_id} has no CLIP embedding")
            return []

        source_embedding = row[0]

        # Find similar images
        result = session.execute(
            text(
                """
                SELECT
                    i.id,
                    i.source_path,
                    1 - (i.clip_embedding <=> :embedding::vector) as similarity
                FROM images i
                WHERE i.catalog_id = :catalog_id
                AND i.id != :source_id
                AND i.clip_embedding IS NOT NULL
                AND 1 - (i.clip_embedding <=> :embedding::vector) >= :threshold
                ORDER BY i.clip_embedding <=> :embedding::vector
                LIMIT :limit
            """
            ),
            {
                "catalog_id": catalog_id,
                "source_id": image_id,
                "embedding": str(list(source_embedding)),
                "threshold": threshold,
                "limit": limit,
            },
        )

        return [
            SearchResult(
                image_id=row[0],
                source_path=row[1],
                similarity_score=float(row[2]),
            )
            for row in result.fetchall()
        ]
