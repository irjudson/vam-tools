"""
Fast similarity search using FAISS.

Provides efficient approximate nearest neighbor search for finding similar
perceptual hashes in large collections.
"""

# mypy: disable-error-code="no-any-return,attr-defined,unreachable"

import logging
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FastSimilaritySearcher:
    """
    Fast similarity search using FAISS index.

    Uses binary hamming distance for perceptual hash comparison.
    """

    def __init__(self, hash_size: int = 64, use_gpu: bool = False):
        """
        Initialize similarity searcher.

        Args:
            hash_size: Size of hash in bits (default 64 for 16-char hex)
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu)
        """
        self.hash_size = hash_size
        self.use_gpu = use_gpu
        self.index = None
        self.id_map: List[str] = []  # Maps index position to image ID

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
