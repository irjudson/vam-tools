"""
Fast similarity search using FAISS.

Provides efficient approximate nearest neighbor search for finding similar
perceptual hashes in large collections.

Supports index persistence to disk for instant startup on subsequent runs.
"""

# mypy: disable-error-code="no-any-return,attr-defined,unreachable"

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Index file format version - increment when format changes
INDEX_FORMAT_VERSION = 1


@dataclass
class IndexMetadata:
    """Metadata for a persisted FAISS index."""

    version: int  # Format version for compatibility checking
    hash_size: int  # Size of hashes in bits (e.g., 64)
    hash_method: str  # Hash method used (e.g., "dhash", "ahash")
    image_count: int  # Number of images in the index
    created_at: str  # ISO timestamp when index was created
    catalog_id: str  # Catalog this index belongs to

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexMetadata":
        """Create from dictionary."""
        return cls(**data)


class FastSimilaritySearcher:
    """
    Fast similarity search using FAISS index.

    Uses binary hamming distance for perceptual hash comparison.
    Supports persistence to disk for instant startup.
    """

    def __init__(
        self,
        hash_size: int = 64,
        use_gpu: bool = False,
        catalog_id: Optional[str] = None,
    ):
        """
        Initialize similarity searcher.

        Args:
            hash_size: Size of hash in bits (default 64 for 16-char hex)
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu)
            catalog_id: Optional catalog ID for index persistence
        """
        self.hash_size = hash_size
        self.use_gpu = use_gpu
        self.catalog_id = catalog_id
        self.index = None
        self.id_map: List[str] = []  # Maps index position to image ID
        self.metadata: Optional[IndexMetadata] = None
        self._hash_method: str = "dhash"  # Track which hash method was used

        try:
            import faiss

            self.faiss = faiss
            self.available = True

            # Check if GPU is available
            if use_gpu and hasattr(faiss, "StandardGpuResources"):
                try:
                    self.gpu_resources = faiss.StandardGpuResources()
                    logger.info("FAISS GPU resources initialized")
                except Exception as e:
                    logger.warning(f"FAISS GPU initialization failed: {e}, using CPU")
                    self.use_gpu = False
            else:
                self.use_gpu = False
                # Optimize FAISS-CPU for multi-core systems
                import multiprocessing

                num_threads = multiprocessing.cpu_count()
                faiss.omp_set_num_threads(num_threads)
                logger.info(f"FAISS-CPU configured to use {num_threads} threads")

        except ImportError:
            logger.warning("FAISS not available, similarity search will be slow")
            self.available = False
            self.faiss = None

    def _hex_to_binary(self, hex_hash: str) -> np.ndarray:
        """
        Convert hex hash string to binary numpy array.

        Args:
            hex_hash: Hexadecimal hash string (e.g., "8f373c3f1f0f0f0f")

        Returns:
            Binary array of shape (hash_size,)
        """
        # Convert hex to integer
        hash_int = int(hex_hash, 16)

        # Convert to binary array
        binary = np.array(
            [(hash_int >> i) & 1 for i in range(self.hash_size)], dtype=np.uint8
        )

        return binary

    def build_index(self, hashes: Dict[str, str], method: str = "dhash") -> None:
        """
        Build FAISS index from hash dictionary.

        Args:
            hashes: Dictionary mapping image IDs to hash strings
            method: Which hash method to use ('dhash', 'ahash', or 'whash')
        """
        if not self.available:
            logger.warning("FAISS not available, skipping index build")
            return

        # Track hash method for persistence
        self._hash_method = method

        # Clear existing index
        self.id_map = []

        # Convert hashes to binary vectors
        vectors = []
        for image_id, hash_value in hashes.items():
            if hash_value:
                vectors.append(self._hex_to_binary(hash_value))
                self.id_map.append(image_id)

        if not vectors:
            logger.warning("No valid hashes to index")
            return

        # Stack vectors
        vectors_np = np.stack(vectors).astype("float32")

        # Create binary index for Hamming distance
        self.index = self.faiss.IndexBinaryFlat(self.hash_size)

        # Move to GPU if requested
        if self.use_gpu:
            try:
                self.index = self.faiss.index_cpu_to_gpu(
                    self.gpu_resources, 0, self.index
                )
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}, using CPU")

        # Add vectors to index (need to pack bits for binary index)
        # Convert float vectors back to uint8 for binary index
        binary_vectors = vectors_np.astype(np.uint8)

        # Pack bits: FAISS binary index expects packed bytes
        # 64 bits = 8 bytes
        packed_vectors = np.packbits(binary_vectors, axis=1)

        self.index.add(packed_vectors)

        # Create metadata for this index
        self.metadata = IndexMetadata(
            version=INDEX_FORMAT_VERSION,
            hash_size=self.hash_size,
            hash_method=method,
            image_count=len(self.id_map),
            created_at=datetime.utcnow().isoformat(),
            catalog_id=self.catalog_id or "unknown",
        )

        logger.info(
            f"Built FAISS index with {len(self.id_map)} hashes "
            f"(GPU: {self.use_gpu})"
        )

    def find_similar(
        self,
        threshold: int = 5,
        k: int = 50,
    ) -> List[Tuple[str, str, int]]:
        """
        Find all pairs of similar images.

        Args:
            threshold: Maximum Hamming distance to consider similar
            k: Number of nearest neighbors to search (higher = slower but more thorough)

        Returns:
            List of (image_id1, image_id2, distance) tuples
        """
        if not self.available or self.index is None:
            logger.warning("FAISS index not available")
            return []

        similar_pairs: Set[Tuple[str, str, int]] = set()

        # Get vectors from index for querying
        n_vectors = self.index.ntotal

        # For binary index, we need to reconstruct vectors for querying
        # Actually, we can just search against the index itself
        logger.info(f"Searching {n_vectors} vectors for similar pairs...")

        # Search in batches for memory efficiency
        batch_size = 1000
        for start_idx in range(0, n_vectors, batch_size):
            end_idx = min(start_idx + batch_size, n_vectors)

            # Get batch of query vectors
            # For binary index, need to reconstruct from packed format
            query_vectors = []
            for idx in range(start_idx, end_idx):
                # Reconstruct vector from index
                # This is a workaround - ideally we'd store original vectors
                query_vectors.append(self.index.reconstruct(idx))

            query_batch = np.stack(query_vectors)

            # Search
            distances, indices = self.index.search(query_batch, k)

            # Process results
            for query_idx, (dists, idxs) in enumerate(zip(distances, indices)):
                global_query_idx = start_idx + query_idx
                query_id = self.id_map[global_query_idx]

                for dist, idx in zip(dists, idxs):
                    # Skip self-matches and pairs beyond threshold
                    if idx == global_query_idx or dist > threshold:
                        continue

                    result_id = self.id_map[idx]

                    # Create sorted tuple to avoid duplicates (a,b) and (b,a)
                    pair = tuple(sorted([query_id, result_id]))
                    similar_pairs.add((*pair, int(dist)))

        logger.info(f"Found {len(similar_pairs)} similar pairs")
        return list(similar_pairs)

    def find_similar_to(
        self,
        query_hash: str,
        threshold: int = 5,
        k: int = 10,
    ) -> List[Tuple[str, int]]:
        """
        Find images similar to a specific hash.

        Args:
            query_hash: Hash string to search for
            threshold: Maximum Hamming distance to consider similar
            k: Maximum number of results to return

        Returns:
            List of (image_id, distance) tuples
        """
        if not self.available or self.index is None:
            return []

        # Convert query hash to binary
        query_vec = self._hex_to_binary(query_hash).astype("float32")

        # Pack for binary index
        query_packed = np.packbits(query_vec.astype(np.uint8))

        # Search
        distances, indices = self.index.search(query_packed.reshape(1, -1), k)

        # Filter by threshold and return
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist <= threshold and idx < len(self.id_map):
                results.append((self.id_map[idx], int(dist)))

        return results

    # =========================================================================
    # Index Persistence Methods
    # =========================================================================

    def get_index_path(self, index_dir: Path) -> Tuple[Path, Path]:
        """
        Get the paths for index and metadata files.

        Args:
            index_dir: Directory to store index files

        Returns:
            Tuple of (index_path, metadata_path)
        """
        catalog_suffix = f"_{self.catalog_id}" if self.catalog_id else ""
        index_path = index_dir / f"faiss_index{catalog_suffix}.bin"
        metadata_path = index_dir / f"faiss_index{catalog_suffix}.json"
        return index_path, metadata_path

    def save(self, index_dir: Path) -> bool:
        """
        Save FAISS index and metadata to disk.

        Args:
            index_dir: Directory to save index files

        Returns:
            True if save was successful, False otherwise
        """
        if not self.available or self.index is None:
            logger.warning("No index to save")
            return False

        if self.metadata is None:
            logger.warning("No metadata available, creating default")
            self.metadata = IndexMetadata(
                version=INDEX_FORMAT_VERSION,
                hash_size=self.hash_size,
                hash_method=self._hash_method,
                image_count=len(self.id_map),
                created_at=datetime.utcnow().isoformat(),
                catalog_id=self.catalog_id or "unknown",
            )

        try:
            # Create directory if needed
            index_dir = Path(index_dir)
            index_dir.mkdir(parents=True, exist_ok=True)

            index_path, metadata_path = self.get_index_path(index_dir)

            # If index is on GPU, move to CPU for saving
            cpu_index = self.index
            if self.use_gpu and hasattr(self.faiss, "index_gpu_to_cpu"):
                cpu_index = self.faiss.index_gpu_to_cpu(self.index)

            # Save FAISS index (binary format)
            self.faiss.write_index_binary(cpu_index, str(index_path))

            # Save metadata and id_map as JSON
            save_data = {
                "metadata": self.metadata.to_dict(),
                "id_map": self.id_map,
            }
            with open(metadata_path, "w") as f:
                json.dump(save_data, f, indent=2)

            logger.info(
                f"Saved FAISS index to {index_path} "
                f"({len(self.id_map)} images, {index_path.stat().st_size / 1024:.1f} KB)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            return False

    def load(self, index_dir: Path) -> bool:
        """
        Load FAISS index and metadata from disk.

        Args:
            index_dir: Directory containing index files

        Returns:
            True if load was successful, False otherwise
        """
        if not self.available:
            logger.warning("FAISS not available, cannot load index")
            return False

        try:
            index_dir = Path(index_dir)
            index_path, metadata_path = self.get_index_path(index_dir)

            if not index_path.exists() or not metadata_path.exists():
                logger.info(f"No saved index found at {index_path}")
                return False

            # Load metadata and id_map
            with open(metadata_path, "r") as f:
                save_data = json.load(f)

            metadata = IndexMetadata.from_dict(save_data["metadata"])

            # Version compatibility check
            if metadata.version > INDEX_FORMAT_VERSION:
                logger.warning(
                    f"Index version {metadata.version} is newer than supported "
                    f"version {INDEX_FORMAT_VERSION}, skipping load"
                )
                return False

            # Hash size compatibility check
            if metadata.hash_size != self.hash_size:
                logger.warning(
                    f"Index hash size {metadata.hash_size} doesn't match "
                    f"expected {self.hash_size}, skipping load"
                )
                return False

            # Load FAISS index
            self.index = self.faiss.read_index_binary(str(index_path))
            self.id_map = save_data["id_map"]
            self.metadata = metadata
            self._hash_method = metadata.hash_method

            # Move to GPU if requested
            if self.use_gpu:
                try:
                    self.index = self.faiss.index_cpu_to_gpu(
                        self.gpu_resources, 0, self.index
                    )
                    logger.info("Loaded index moved to GPU")
                except Exception as e:
                    logger.warning(f"Failed to move loaded index to GPU: {e}")

            logger.info(
                f"Loaded FAISS index from {index_path} "
                f"({len(self.id_map)} images, hash_method={metadata.hash_method})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def is_valid_for(
        self,
        expected_count: int,
        expected_method: str = "dhash",
        tolerance: float = 0.1,
    ) -> bool:
        """
        Check if the loaded index is valid for the current catalog state.

        Args:
            expected_count: Expected number of images in catalog
            expected_method: Expected hash method
            tolerance: Allowed percentage difference in image count (0.1 = 10%)

        Returns:
            True if index is valid and usable, False if rebuild is needed
        """
        if self.index is None or self.metadata is None:
            return False

        # Check hash method matches
        if self.metadata.hash_method != expected_method:
            logger.info(
                f"Index hash method {self.metadata.hash_method} != {expected_method}"
            )
            return False

        # Check image count is within tolerance
        actual_count = len(self.id_map)
        if expected_count == 0:
            return actual_count == 0

        diff_ratio = abs(actual_count - expected_count) / expected_count
        if diff_ratio > tolerance:
            logger.info(
                f"Index image count {actual_count} differs from expected "
                f"{expected_count} by {diff_ratio:.1%} (tolerance: {tolerance:.1%})"
            )
            return False

        return True

    def needs_rebuild(
        self,
        current_image_ids: Set[str],
        tolerance: float = 0.1,
    ) -> Tuple[bool, Set[str], Set[str]]:
        """
        Determine if index needs rebuild based on current image set.

        Args:
            current_image_ids: Set of image IDs currently in catalog
            tolerance: Allowed percentage of missing/extra images

        Returns:
            Tuple of (needs_rebuild, missing_ids, extra_ids)
        """
        if self.index is None:
            return True, current_image_ids, set()

        indexed_ids = set(self.id_map)
        missing_ids = current_image_ids - indexed_ids  # In catalog, not in index
        extra_ids = indexed_ids - current_image_ids  # In index, not in catalog

        total_diff = len(missing_ids) + len(extra_ids)
        if len(current_image_ids) == 0:
            diff_ratio = 1.0 if total_diff > 0 else 0.0
        else:
            diff_ratio = total_diff / len(current_image_ids)

        needs_rebuild = diff_ratio > tolerance

        if needs_rebuild:
            logger.info(
                f"Index needs rebuild: {len(missing_ids)} missing, "
                f"{len(extra_ids)} extra ({diff_ratio:.1%} diff)"
            )

        return needs_rebuild, missing_ids, extra_ids

    def add_hashes(self, new_hashes: Dict[str, str]) -> int:
        """
        Add new hashes to the existing index (incremental update).

        Args:
            new_hashes: Dictionary mapping new image IDs to hash strings

        Returns:
            Number of hashes added
        """
        if not self.available or self.index is None:
            logger.warning("Cannot add hashes: no index available")
            return 0

        # Filter out hashes that are already in the index
        existing_ids = set(self.id_map)
        hashes_to_add = {
            k: v for k, v in new_hashes.items() if k not in existing_ids and v
        }

        if not hashes_to_add:
            return 0

        # Convert hashes to binary vectors
        vectors = []
        new_ids = []
        for image_id, hash_value in hashes_to_add.items():
            vectors.append(self._hex_to_binary(hash_value))
            new_ids.append(image_id)

        # Stack and pack vectors
        vectors_np = np.stack(vectors).astype(np.uint8)
        packed_vectors = np.packbits(vectors_np, axis=1)

        # Add to index
        self.index.add(packed_vectors)
        self.id_map.extend(new_ids)

        # Update metadata
        if self.metadata:
            self.metadata.image_count = len(self.id_map)

        logger.info(f"Added {len(new_ids)} hashes to index (total: {len(self.id_map)})")
        return len(new_ids)

    def remove_ids(self, ids_to_remove: Set[str]) -> int:
        """
        Mark IDs for removal from the index.

        Note: FAISS IndexBinaryFlat doesn't support removal, so this marks
        them as invalid. The index should be rebuilt periodically to reclaim space.

        Args:
            ids_to_remove: Set of image IDs to remove

        Returns:
            Number of IDs marked for removal
        """
        if not self.id_map:
            return 0

        # We can't actually remove from FAISS index, but we can mark as removed
        # by setting the ID to a special marker
        removed_count = 0
        for i, image_id in enumerate(self.id_map):
            if image_id in ids_to_remove:
                self.id_map[i] = "__REMOVED__"
                removed_count += 1

        if removed_count > 0:
            logger.info(
                f"Marked {removed_count} IDs for removal "
                "(rebuild index to reclaim space)"
            )

        return removed_count

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.

        Returns:
            Dictionary with index statistics
        """
        stats: Dict[str, Any] = {
            "available": self.available,
            "has_index": self.index is not None,
            "use_gpu": self.use_gpu,
            "hash_size": self.hash_size,
        }

        if self.index is not None:
            stats["total_vectors"] = self.index.ntotal
            stats["id_map_size"] = len(self.id_map)
            removed = sum(1 for x in self.id_map if x == "__REMOVED__")
            stats["removed_count"] = removed
            stats["active_count"] = len(self.id_map) - removed

        if self.metadata:
            stats["metadata"] = self.metadata.to_dict()

        return stats
