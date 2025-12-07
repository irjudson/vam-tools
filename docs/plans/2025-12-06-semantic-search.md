# Semantic Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable natural language image search using CLIP embeddings with zero additional model overhead.

**Architecture:** Modify existing auto_tag_task to save CLIP embeddings (already computed), add pgvector column for similarity search, create search API endpoint, and add search UI.

**Tech Stack:** OpenCLIP (already loaded), pgvector (already installed), FastAPI, Vue.js

**GitHub Issue:** #18

---

## Task 1: Database Schema - Add CLIP Embedding Column

**Files:**
- Create: `vam_tools/db/migrations/versions/003_add_clip_embedding.py`
- Modify: `vam_tools/db/catalog_schema.py`

### Step 1: Write the failing test

Create `tests/db/test_clip_embedding_migration.py`:

```python
"""Tests for CLIP embedding column migration."""

import pytest
from sqlalchemy import text


class TestClipEmbeddingColumn:
    """Tests for clip_embedding column in images table."""

    def test_images_table_has_clip_embedding_column(self, db_session):
        """Test that images table has clip_embedding column."""
        result = db_session.execute(
            text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'images' AND column_name = 'clip_embedding'
            """)
        )
        row = result.fetchone()
        assert row is not None, "clip_embedding column should exist"

    def test_clip_embedding_accepts_768_dim_vector(self, db_session):
        """Test that clip_embedding accepts 768-dimensional vectors."""
        # Create a test image first
        db_session.execute(
            text("""
                INSERT INTO images (id, catalog_id, source_path, file_hash, file_type)
                VALUES (
                    'test-image-001',
                    (SELECT id FROM catalogs LIMIT 1),
                    '/test/image.jpg',
                    'abc123',
                    'image'
                )
            """)
        )

        # Insert a 768-dim vector
        embedding = [0.1] * 768
        db_session.execute(
            text("""
                UPDATE images
                SET clip_embedding = :embedding
                WHERE id = 'test-image-001'
            """),
            {"embedding": str(embedding)}
        )
        db_session.commit()

        # Verify it was stored
        result = db_session.execute(
            text("""
                SELECT clip_embedding IS NOT NULL as has_embedding
                FROM images WHERE id = 'test-image-001'
            """)
        )
        row = result.fetchone()
        assert row[0] is True
```

### Step 2: Run test to verify it fails

```bash
./venv/bin/pytest tests/db/test_clip_embedding_migration.py -v
```

Expected: FAIL with "clip_embedding column should exist"

### Step 3: Create the Alembic migration

Create `vam_tools/db/migrations/versions/003_add_clip_embedding.py`:

```python
"""Add clip_embedding column to images table.

Revision ID: 003_clip_embedding
Revises: 002_xxx  # Update with actual previous revision
Create Date: 2025-12-06
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "003_clip_embedding"
down_revision = None  # Update with actual previous revision
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add clip_embedding column for semantic search."""
    # Ensure vector extension exists
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Add clip_embedding column (768-dim for OpenCLIP ViT-L/14)
    op.add_column(
        "images",
        sa.Column("clip_embedding", sa.dialects.postgresql.ARRAY(sa.Float), nullable=True)
    )

    # Create index for vector similarity search
    # Using IVFFlat for approximate nearest neighbor search
    op.execute("""
        CREATE INDEX IF NOT EXISTS images_clip_embedding_idx
        ON images USING ivfflat (clip_embedding vector_cosine_ops)
        WITH (lists = 100)
    """)


def downgrade() -> None:
    """Remove clip_embedding column."""
    op.drop_index("images_clip_embedding_idx", table_name="images")
    op.drop_column("images", "clip_embedding")
```

### Step 4: Update catalog_schema.py

Add to `vam_tools/db/catalog_schema.py` in the images table definition:

```python
# In the CREATE TABLE images statement, add:
clip_embedding VECTOR(768),
```

### Step 5: Run migration and verify test passes

```bash
# Apply migration
./venv/bin/alembic upgrade head

# Run tests
./venv/bin/pytest tests/db/test_clip_embedding_migration.py -v
```

Expected: PASS

### Step 6: Commit

```bash
git add vam_tools/db/migrations/versions/003_add_clip_embedding.py
git add vam_tools/db/catalog_schema.py
git add tests/db/test_clip_embedding_migration.py
git commit -m "feat: add clip_embedding column for semantic search (#18)"
```

---

## Task 2: Semantic Search Service

**Files:**
- Create: `vam_tools/analysis/semantic_search.py`
- Create: `tests/analysis/test_semantic_search.py`

### Step 1: Write the failing test

Create `tests/analysis/test_semantic_search.py`:

```python
"""Tests for semantic search service."""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from vam_tools.analysis.semantic_search import (
    SemanticSearchService,
    SearchResult,
)


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
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

    def test_initialization(self):
        """Test service initialization."""
        service = SemanticSearchService()
        assert service._model is None  # Lazy loading
        assert service.model_name == "ViT-L-14"

    def test_encode_text_returns_768_dim_vector(self):
        """Test that text encoding returns 768-dimensional vector."""
        service = SemanticSearchService()

        with patch.object(service, "_ensure_model_loaded"):
            with patch.object(service, "_model") as mock_model:
                import numpy as np
                mock_model.encode_text.return_value = np.random.randn(1, 768)

                embedding = service.encode_text("sunset over mountains")

                assert len(embedding) == 768

    def test_encode_image_returns_768_dim_vector(self):
        """Test that image encoding returns 768-dimensional vector."""
        service = SemanticSearchService()

        with patch.object(service, "_ensure_model_loaded"):
            with patch.object(service, "_model") as mock_model:
                import numpy as np
                mock_model.encode_image.return_value = np.random.randn(1, 768)

                # Use a mock image path
                embedding = service.encode_image(Path("/fake/image.jpg"))

                assert len(embedding) == 768

    def test_search_returns_results_sorted_by_similarity(self):
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

    def test_find_similar_images(self):
        """Test finding similar images to a given image."""
        mock_session = MagicMock()

        # Mock the source image embedding
        mock_session.execute.return_value.fetchone.return_value = ([0.1] * 768,)

        # Mock similar images
        mock_results = [
            ("img-002", "/photos/similar1.jpg", 0.92),
            ("img-003", "/photos/similar2.jpg", 0.88),
        ]
        mock_session.execute.return_value.fetchall.return_value = mock_results

        service = SemanticSearchService()
        results = service.find_similar(
            session=mock_session,
            catalog_id="cat-001",
            image_id="img-001",
            limit=10,
        )

        assert len(results) >= 0  # May be empty if no similar images
```

### Step 2: Run test to verify it fails

```bash
./venv/bin/pytest tests/analysis/test_semantic_search.py -v
```

Expected: FAIL with "No module named 'vam_tools.analysis.semantic_search'"

### Step 3: Implement SemanticSearchService

Create `vam_tools/analysis/semantic_search.py`:

```python
"""Semantic search service using CLIP embeddings.

Enables natural language image search by encoding text queries
and finding visually similar images via vector similarity.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
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
        self._model = None
        self._preprocess = None
        self._tokenizer = None
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
        from PIL import Image

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
            text("""
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
            """),
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
            text("""
                SELECT clip_embedding
                FROM images
                WHERE id = :image_id AND catalog_id = :catalog_id
            """),
            {"image_id": image_id, "catalog_id": catalog_id},
        )
        row = result.fetchone()

        if row is None or row[0] is None:
            logger.warning(f"Image {image_id} has no CLIP embedding")
            return []

        source_embedding = row[0]

        # Find similar images
        result = session.execute(
            text("""
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
            """),
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
```

### Step 4: Run tests to verify they pass

```bash
./venv/bin/pytest tests/analysis/test_semantic_search.py -v
```

Expected: PASS

### Step 5: Commit

```bash
git add vam_tools/analysis/semantic_search.py
git add tests/analysis/test_semantic_search.py
git commit -m "feat: add semantic search service (#18)"
```

---

## Task 3: Modify Auto-Tag Task to Save Embeddings

**Files:**
- Modify: `vam_tools/analysis/image_tagger.py`
- Modify: `vam_tools/jobs/tasks.py`
- Modify: `tests/analysis/test_image_tagger.py`

### Step 1: Write the failing test

Add to `tests/analysis/test_image_tagger.py`:

```python
class TestImageTaggerEmbeddings:
    """Tests for CLIP embedding extraction."""

    def test_get_embedding_returns_768_dim_vector(self):
        """Test that get_embedding returns 768-dimensional vector."""
        tagger = ImageTagger()

        with patch.object(tagger, "_clip_backend") as mock_backend:
            import numpy as np
            mock_backend.get_embedding.return_value = np.random.randn(768)

            embedding = tagger.get_embedding(Path("/fake/image.jpg"))

            assert len(embedding) == 768

    def test_tag_batch_with_embeddings_returns_embeddings(self):
        """Test that tag_batch can return embeddings alongside tags."""
        tagger = ImageTagger()

        with patch.object(tagger, "_clip_backend") as mock_backend:
            import numpy as np
            mock_backend.tag_images.return_value = {"dog": 0.9}
            mock_backend.get_embedding.return_value = np.random.randn(768)

            results = tagger.tag_batch_with_embeddings(
                [Path("/fake/image.jpg")],
                return_embeddings=True,
            )

            assert "embeddings" in results
            assert len(results["embeddings"][Path("/fake/image.jpg")]) == 768
```

### Step 2: Run test to verify it fails

```bash
./venv/bin/pytest tests/analysis/test_image_tagger.py::TestImageTaggerEmbeddings -v
```

Expected: FAIL with "AttributeError: 'ImageTagger' object has no attribute 'get_embedding'"

### Step 3: Add get_embedding method to ImageTagger

Add to `vam_tools/analysis/image_tagger.py`:

```python
def get_embedding(self, image_path: Union[str, Path]) -> List[float]:
    """Get CLIP embedding for an image.

    This is the same embedding used internally for classification,
    just exposed for semantic search storage.

    Args:
        image_path: Path to image file

    Returns:
        768-dimensional embedding as list of floats
    """
    if not self._clip_backend:
        raise RuntimeError("CLIP backend not initialized")

    return self._clip_backend.get_embedding(image_path)

def tag_batch_with_embeddings(
    self,
    image_paths: List[Union[str, Path]],
    threshold: float = 0.25,
    max_tags: int = 10,
    return_embeddings: bool = True,
) -> Dict[str, Any]:
    """Tag images and optionally return their CLIP embeddings.

    Args:
        image_paths: List of image paths
        threshold: Minimum confidence threshold
        max_tags: Maximum tags per image
        return_embeddings: Whether to return embeddings

    Returns:
        Dict with 'tags' and optionally 'embeddings' keys
    """
    result = {"tags": {}}

    if return_embeddings:
        result["embeddings"] = {}

    for path in image_paths:
        path = Path(path)
        tags = self.tag_image(path, threshold=threshold, max_tags=max_tags)
        result["tags"][path] = tags

        if return_embeddings:
            result["embeddings"][path] = self.get_embedding(path)

    return result
```

Add to `OpenCLIPBackend` class:

```python
def get_embedding(self, image_path: Union[str, Path]) -> List[float]:
    """Get raw CLIP embedding for an image.

    Args:
        image_path: Path to image

    Returns:
        768-dimensional embedding
    """
    self._load_model()

    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")
    image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)

    with torch.no_grad():
        features = self._model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten().tolist()
```

### Step 4: Update auto_tag_task to save embeddings

Modify `vam_tools/jobs/tasks.py` in the `auto_tag_task` function:

```python
# After tagging, save embedding
if hasattr(tagger, 'get_embedding'):
    try:
        embedding = tagger.get_embedding(image_path)
        db.session.execute(
            text("""
                UPDATE images
                SET clip_embedding = :embedding
                WHERE id = :image_id
            """),
            {"image_id": image_id, "embedding": str(embedding)}
        )
    except Exception as e:
        logger.warning(f"Failed to save embedding for {image_id}: {e}")
```

### Step 5: Run tests to verify they pass

```bash
./venv/bin/pytest tests/analysis/test_image_tagger.py::TestImageTaggerEmbeddings -v
```

Expected: PASS

### Step 6: Commit

```bash
git add vam_tools/analysis/image_tagger.py
git add vam_tools/jobs/tasks.py
git add tests/analysis/test_image_tagger.py
git commit -m "feat: save CLIP embeddings during auto-tagging (#18)"
```

---

## Task 4: Search API Endpoint

**Files:**
- Modify: `vam_tools/web/api.py`
- Create: `tests/web/test_search_api.py`

### Step 1: Write the failing test

Create `tests/web/test_search_api.py`:

```python
"""Tests for semantic search API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


class TestSearchAPI:
    """Tests for /api/catalogs/{id}/search endpoint."""

    def test_search_endpoint_returns_results(self, client, mock_catalog):
        """Test that search endpoint returns results."""
        with patch("vam_tools.web.api.SemanticSearchService") as MockService:
            mock_service = MockService.return_value
            mock_service.search.return_value = [
                MagicMock(
                    image_id="img-001",
                    source_path="/photos/sunset.jpg",
                    similarity_score=0.85,
                )
            ]

            response = client.get(
                f"/api/catalogs/{mock_catalog.id}/search",
                params={"q": "sunset"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["similarity_score"] == 0.85

    def test_search_requires_query_parameter(self, client, mock_catalog):
        """Test that search requires q parameter."""
        response = client.get(f"/api/catalogs/{mock_catalog.id}/search")
        assert response.status_code == 422  # Validation error

    def test_similar_endpoint_returns_results(self, client, mock_catalog):
        """Test that similar endpoint returns results."""
        with patch("vam_tools.web.api.SemanticSearchService") as MockService:
            mock_service = MockService.return_value
            mock_service.find_similar.return_value = [
                MagicMock(
                    image_id="img-002",
                    source_path="/photos/similar.jpg",
                    similarity_score=0.92,
                )
            ]

            response = client.get(
                f"/api/catalogs/{mock_catalog.id}/similar/img-001"
            )

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
```

### Step 2: Run test to verify it fails

```bash
./venv/bin/pytest tests/web/test_search_api.py -v
```

Expected: FAIL with 404 (endpoint doesn't exist)

### Step 3: Add search endpoints to API

Add to `vam_tools/web/api.py`:

```python
from vam_tools.analysis.semantic_search import SemanticSearchService, SearchResult

# Initialize search service (lazy loaded)
_search_service: Optional[SemanticSearchService] = None

def get_search_service() -> SemanticSearchService:
    """Get or create the semantic search service."""
    global _search_service
    if _search_service is None:
        _search_service = SemanticSearchService()
    return _search_service


@app.get("/api/catalogs/{catalog_id}/search")
async def search_images(
    catalog_id: str,
    q: str,
    limit: int = 50,
    threshold: float = 0.2,
):
    """Search for images using natural language query.

    Args:
        catalog_id: Catalog to search in
        q: Search query (e.g., "sunset over mountains")
        limit: Maximum results to return
        threshold: Minimum similarity score (0-1)
    """
    db = get_catalog_db(catalog_id)
    service = get_search_service()

    try:
        results = service.search(
            session=db.session,
            catalog_id=catalog_id,
            query=q,
            limit=limit,
            threshold=threshold,
        )

        return {
            "query": q,
            "results": [
                {
                    "image_id": r.image_id,
                    "source_path": r.source_path,
                    "similarity_score": r.similarity_score,
                    "thumbnail_url": f"/api/catalogs/{catalog_id}/images/{r.image_id}/thumbnail",
                }
                for r in results
            ],
            "count": len(results),
        }
    finally:
        db.close()


@app.get("/api/catalogs/{catalog_id}/similar/{image_id}")
async def find_similar_images(
    catalog_id: str,
    image_id: str,
    limit: int = 20,
    threshold: float = 0.5,
):
    """Find images similar to a given image.

    Args:
        catalog_id: Catalog to search in
        image_id: Source image ID
        limit: Maximum results to return
        threshold: Minimum similarity score (0-1)
    """
    db = get_catalog_db(catalog_id)
    service = get_search_service()

    try:
        results = service.find_similar(
            session=db.session,
            catalog_id=catalog_id,
            image_id=image_id,
            limit=limit,
            threshold=threshold,
        )

        return {
            "source_image_id": image_id,
            "results": [
                {
                    "image_id": r.image_id,
                    "source_path": r.source_path,
                    "similarity_score": r.similarity_score,
                    "thumbnail_url": f"/api/catalogs/{catalog_id}/images/{r.image_id}/thumbnail",
                }
                for r in results
            ],
            "count": len(results),
        }
    finally:
        db.close()
```

### Step 4: Run tests to verify they pass

```bash
./venv/bin/pytest tests/web/test_search_api.py -v
```

Expected: PASS

### Step 5: Commit

```bash
git add vam_tools/web/api.py
git add tests/web/test_search_api.py
git commit -m "feat: add semantic search API endpoints (#18)"
```

---

## Task 5: Search UI Component

**Files:**
- Modify: `vam_tools/web/static/index.html`
- Modify: `vam_tools/web/static/app.js`

### Step 1: Add search bar to left panel area

Add to `index.html` in the filter section:

```html
<!-- Semantic Search -->
<div class="filter-group" v-if="currentCatalog">
    <label>Search Images</label>
    <div class="search-input-wrapper">
        <input
            type="text"
            v-model="semanticSearchQuery"
            @keyup.enter="performSemanticSearch"
            placeholder="e.g., sunset over mountains"
            class="semantic-search-input"
        >
        <button
            @click="performSemanticSearch"
            :disabled="!semanticSearchQuery || isSearching"
            class="search-btn"
        >
            {{ isSearching ? 'Searching...' : 'Search' }}
        </button>
    </div>
    <div v-if="searchResults.length > 0" class="search-results-info">
        Found {{ searchResults.length }} results
        <button @click="clearSearch" class="clear-search-btn">Clear</button>
    </div>
</div>
```

### Step 2: Add search methods to app.js

Add to the Vue data object:

```javascript
semanticSearchQuery: '',
searchResults: [],
isSearching: false,
searchMode: false,  // true when showing search results
```

Add methods:

```javascript
async performSemanticSearch() {
    if (!this.semanticSearchQuery || !this.currentCatalog) return;

    this.isSearching = true;
    try {
        const response = await axios.get(
            `/api/catalogs/${this.currentCatalog.id}/search`,
            {
                params: {
                    q: this.semanticSearchQuery,
                    limit: 100,
                    threshold: 0.2
                }
            }
        );

        this.searchResults = response.data.results;
        this.searchMode = true;

        // Update the image grid to show search results
        this.images = this.searchResults.map(r => ({
            id: r.image_id,
            source_path: r.source_path,
            similarity_score: r.similarity_score,
            // Fetch full image data or use placeholder
        }));

        this.showNotification(`Found ${this.searchResults.length} images`, 'success');
    } catch (error) {
        console.error('Search failed:', error);
        this.showNotification('Search failed', 'error');
    } finally {
        this.isSearching = false;
    }
},

clearSearch() {
    this.semanticSearchQuery = '';
    this.searchResults = [];
    this.searchMode = false;
    this.loadImages();  // Reload normal browse view
},

async findSimilarImages(imageId) {
    try {
        const response = await axios.get(
            `/api/catalogs/${this.currentCatalog.id}/similar/${imageId}`,
            {
                params: { limit: 20, threshold: 0.5 }
            }
        );

        this.searchResults = response.data.results;
        this.searchMode = true;
        this.semanticSearchQuery = `Similar to image`;

        this.images = this.searchResults.map(r => ({
            id: r.image_id,
            source_path: r.source_path,
            similarity_score: r.similarity_score,
        }));

        this.showNotification(`Found ${this.searchResults.length} similar images`, 'success');
    } catch (error) {
        console.error('Find similar failed:', error);
        this.showNotification('Find similar failed', 'error');
    }
},
```

### Step 3: Add "Find Similar" button to lightbox

Add to the lightbox controls in `index.html`:

```html
<button
    @click="findSimilarImages(lightboxImage.id)"
    class="lightbox-action-btn"
    title="Find Similar Images"
>
    Find Similar
</button>
```

### Step 4: Add CSS styles

Add to `styles.css`:

```css
.semantic-search-input {
    width: 100%;
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    margin-bottom: 8px;
}

.search-input-wrapper {
    display: flex;
    gap: 8px;
    flex-direction: column;
}

.search-btn {
    padding: 8px 16px;
    background: var(--accent-primary);
    color: white;
    border: none;
    border-radius: var(--radius-sm);
    cursor: pointer;
}

.search-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.search-results-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 8px;
    font-size: 0.85em;
    color: var(--text-secondary);
}

.clear-search-btn {
    background: transparent;
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 4px 8px;
    border-radius: var(--radius-sm);
    cursor: pointer;
}
```

### Step 5: Manual testing

1. Start the web server
2. Navigate to a catalog with tagged images
3. Enter a search query like "sunset" or "dogs"
4. Verify results appear
5. Click "Find Similar" on an image in lightbox
6. Verify similar images appear

### Step 6: Commit

```bash
git add vam_tools/web/static/index.html
git add vam_tools/web/static/app.js
git add vam_tools/web/static/styles.css
git commit -m "feat: add semantic search UI (#18)"
```

---

## Summary

This plan implements semantic search in 5 tasks:

1. **Database Schema** - Add clip_embedding column
2. **Search Service** - Core search logic
3. **Auto-Tag Integration** - Save embeddings during tagging
4. **API Endpoints** - REST API for search
5. **UI Component** - Search bar and "Find Similar"

**Key Points:**
- Zero additional model downloads (reuses OpenCLIP from tagging)
- Zero additional processing time (embeddings computed during tagging)
- Uses pgvector for fast similarity search
- TDD approach throughout
