"""
Duplicate detection system.

Identifies duplicate and similar images using:
1. Exact duplicates (same checksum)
2. Near duplicates (similar perceptual hash)
3. Quality scoring to select the best copy
"""

import logging
from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, List, Optional, Set

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from ..core.catalog import CatalogDatabase
from ..core.performance_stats import PerformanceTracker
from ..core.types import DuplicateGroup, ImageRecord, SimilarityMetrics
from .perceptual_hash import (
    HashMethod,
    combined_hash,
    hamming_distance,
    similarity_score,
)
from .quality_scorer import calculate_quality_score, select_best

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """Detect and manage duplicate images in a catalog."""

    def __init__(
        self,
        catalog: CatalogDatabase,
        similarity_threshold: int = 5,
        hash_size: int = 8,
        hash_methods: Optional[List[HashMethod]] = None,
        use_gpu: bool = False,
        gpu_batch_size: Optional[int] = None,
        use_faiss: bool = False,
        perf_tracker: Optional[PerformanceTracker] = None,
    ):
        """
        Initialize duplicate detector.

        Args:
            catalog: Catalog database to analyze
            similarity_threshold: Maximum Hamming distance for similar images (default: 5)
            hash_size: Size of perceptual hash (default: 8 = 64-bit)
            hash_methods: Hash methods to use (default: [DHASH, AHASH, WHASH])
            use_gpu: Enable GPU acceleration for hash computation
            gpu_batch_size: Batch size for GPU processing (None = auto)
            use_faiss: Enable FAISS for fast similarity search
            perf_tracker: Optional performance tracker for collecting metrics
        """
        self.catalog = catalog
        self.similarity_threshold = similarity_threshold
        self.hash_size = hash_size
        self.hash_methods = hash_methods or [
            HashMethod.DHASH,
            HashMethod.AHASH,
            HashMethod.WHASH,
        ]
        self.use_gpu = use_gpu
        self.gpu_batch_size = gpu_batch_size
        self.use_faiss = use_faiss
        self.duplicate_groups: List[DuplicateGroup] = []
        self.perf_tracker = perf_tracker

    def detect_duplicates(self, recompute_hashes: bool = False) -> List[DuplicateGroup]:
        """
        Detect all duplicate groups in the catalog.

        Process:
        1. Compute perceptual hashes for all images (if not cached)
        2. Find exact duplicates (same checksum)
        3. Find similar images (similar perceptual hash)
        4. Score quality for each image
        5. Select primary (best quality) in each group

        Args:
            recompute_hashes: Force recomputation of perceptual hashes

        Returns:
            List of duplicate groups found
        """
        logger.info("Starting duplicate detection")

        # Track overall duplicate detection
        detect_ctx = (
            self.perf_tracker.track_operation("duplicate_detection")
            if self.perf_tracker
            else nullcontext()
        )

        with detect_ctx:
            # Step 1: Compute perceptual hashes
            compute_ctx = (
                self.perf_tracker.track_operation("compute_hashes")
                if self.perf_tracker
                else nullcontext()
            )
            with compute_ctx:
                self._compute_perceptual_hashes(recompute_hashes)

            # Step 2: Find exact duplicates (same checksum)
            exact_ctx = (
                self.perf_tracker.track_operation("find_exact_duplicates")
                if self.perf_tracker
                else nullcontext()
            )
            with exact_ctx:
                exact_groups = self._find_exact_duplicates()
            logger.info(f"Found {len(exact_groups)} exact duplicate groups")

            # Step 3: Find similar images (perceptual hash similarity)
            similar_ctx = (
                self.perf_tracker.track_operation(
                    "find_similar_images", items=len(self.catalog.list_images())
                )
                if self.perf_tracker
                else nullcontext()
            )
            with similar_ctx:
                similar_groups = self._find_similar_images()
            logger.info(f"Found {len(similar_groups)} similar image groups")

            # Step 4: Merge and deduplicate groups
            merge_ctx = (
                self.perf_tracker.track_operation("merge_groups")
                if self.perf_tracker
                else nullcontext()
            )
            with merge_ctx:
                all_groups = self._merge_duplicate_groups(exact_groups, similar_groups)
            logger.info(f"Total {len(all_groups)} duplicate groups after merging")

            # Step 5: Score quality and select primary for each group
            score_ctx = (
                self.perf_tracker.track_operation(
                    "score_quality", items=len(all_groups)
                )
                if self.perf_tracker
                else nullcontext()
            )
            with score_ctx:
                self._score_and_select_primary(all_groups)

            self.duplicate_groups = all_groups
            return all_groups

    def _compute_perceptual_hashes(self, force: bool = False) -> None:
        """
        Compute perceptual hashes for all images in catalog.

        Args:
            force: Force recomputation even if hash exists
        """
        images = self.catalog.list_images()
        images_to_process = []

        for image in images:
            # Check if we need to compute hash
            # Need to recompute if any of the requested hash methods are missing
            needs_compute = force
            if not force:
                for method in self.hash_methods:
                    if (
                        method == HashMethod.DHASH
                        and not image.metadata.perceptual_hash_dhash
                    ):
                        needs_compute = True
                    elif (
                        method == HashMethod.AHASH
                        and not image.metadata.perceptual_hash_ahash
                    ):
                        needs_compute = True
                    elif (
                        method == HashMethod.WHASH
                        and not image.metadata.perceptual_hash_whash
                    ):
                        needs_compute = True

            if needs_compute:
                images_to_process.append(image)

        if not images_to_process:
            logger.info("All images already have perceptual hashes")
            return

        logger.info(f"Computing perceptual hashes for {len(images_to_process)} images")

        # Use GPU batch processing if enabled
        if self.use_gpu:
            from .perceptual_hash import compute_hashes_batch

            image_paths = [img.source_path for img in images_to_process]

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    "Computing hashes (GPU)...", total=len(images_to_process)
                )

                # Process all images in batches
                all_hashes = compute_hashes_batch(
                    image_paths,
                    use_gpu=True,
                    batch_size=self.gpu_batch_size,
                )

                # Update catalog with computed hashes
                for image, hashes in zip(images_to_process, all_hashes):
                    if hashes:
                        if "dhash" in hashes:
                            image.metadata.perceptual_hash_dhash = hashes["dhash"]
                        if "ahash" in hashes:
                            image.metadata.perceptual_hash_ahash = hashes["ahash"]
                        if "whash" in hashes:
                            image.metadata.perceptual_hash_whash = hashes["whash"]
                        self.catalog.update_image(image)
                    else:
                        logger.warning(
                            f"Failed to compute hash for {image.source_path}"
                        )
                    progress.advance(task)

        else:
            # CPU fallback - sequential processing
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    "Computing hashes...", total=len(images_to_process)
                )

                for image in images_to_process:
                    # Compute requested hashes
                    hashes = combined_hash(
                        image.source_path, self.hash_size, self.hash_methods
                    )

                    if hashes:
                        # Update image metadata with computed hashes
                        if "dhash" in hashes:
                            image.metadata.perceptual_hash_dhash = hashes["dhash"]
                        if "ahash" in hashes:
                            image.metadata.perceptual_hash_ahash = hashes["ahash"]
                        if "whash" in hashes:
                            image.metadata.perceptual_hash_whash = hashes["whash"]
                        # Save to catalog
                        self.catalog.update_image(image)
                    else:
                        logger.warning(
                            f"Failed to compute hash for {image.source_path}"
                        )

                    progress.advance(task)

        # Save catalog after hash computation
        self.catalog.save()
        logger.info("Perceptual hash computation complete")

    def _find_exact_duplicates(self) -> List[DuplicateGroup]:
        """
        Find exact duplicate groups (same checksum).

        Returns:
            List of duplicate groups
        """
        # Group images by checksum
        checksum_groups: Dict[str, List[str]] = defaultdict(list)

        for image in self.catalog.list_images():
            checksum_groups[image.checksum].append(image.id)

        # Create duplicate groups for checksums with multiple images
        groups = []
        for checksum, image_ids in checksum_groups.items():
            if len(image_ids) > 1:
                group = DuplicateGroup(
                    id=f"exact_{checksum[:16]}",
                    images=image_ids,
                    perceptual_hash=None,  # Exact duplicates don't need hash
                )
                groups.append(group)

        return groups

    def _find_similar_images(self) -> List[DuplicateGroup]:
        """
        Find similar image groups using perceptual hashing.

        Returns:
            List of similar image groups
        """
        images = self.catalog.list_images()

        # Filter to images with at least one valid perceptual hash
        # Images must have hashes for the methods we're using
        hashed_images = []
        for img in images:
            has_required_hashes = True
            for method in self.hash_methods:
                if (
                    method == HashMethod.DHASH
                    and not img.metadata.perceptual_hash_dhash
                ):
                    has_required_hashes = False
                    break
                elif (
                    method == HashMethod.AHASH
                    and not img.metadata.perceptual_hash_ahash
                ):
                    has_required_hashes = False
                    break
                elif (
                    method == HashMethod.WHASH
                    and not img.metadata.perceptual_hash_whash
                ):
                    has_required_hashes = False
                    break
            if has_required_hashes:
                hashed_images.append(img)

        if not hashed_images:
            logger.warning("No images with perceptual hashes found")
            return []

        logger.info(
            f"Comparing {len(hashed_images)} images for similarity "
            f"(threshold: {self.similarity_threshold})"
        )

        # Use FAISS for fast similarity search if enabled
        if self.use_faiss:
            return self._find_similar_with_faiss(hashed_images)

        # Fallback to pairwise comparison
        # Track which images have been grouped
        grouped: Set[str] = set()
        groups: List[DuplicateGroup] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "Finding similar images...", total=len(hashed_images)
            )

            for i, image1 in enumerate(hashed_images):
                if image1.id in grouped:
                    progress.advance(task)
                    continue

                # Find all similar images
                similar_ids = [image1.id]
                similarity_metrics = {}

                for image2 in hashed_images[i + 1 :]:
                    if image2.id in grouped:
                        continue

                    # Compare perceptual hashes
                    if self._are_images_similar(image1, image2):
                        similar_ids.append(image2.id)
                        grouped.add(image2.id)

                        # Capture similarity metrics for this pair
                        metrics = self._compute_similarity_metrics(image1, image2)
                        if metrics:
                            # Use sorted IDs for consistent key format
                            key = self._make_pair_key(image1.id, image2.id)
                            similarity_metrics[key] = metrics

                # Create group if we found similar images
                if len(similar_ids) > 1:
                    group = DuplicateGroup(
                        id=f"similar_{image1.id[:16]}",
                        images=similar_ids,
                        perceptual_hash=image1.metadata.perceptual_hash_dhash,
                        similarity_metrics=similarity_metrics,
                    )
                    groups.append(group)
                    grouped.add(image1.id)

                progress.advance(task)

        return groups

    def _find_similar_with_faiss(
        self, hashed_images: List[ImageRecord]
    ) -> List[DuplicateGroup]:
        """
        Find similar images using FAISS fast similarity search.

        Args:
            hashed_images: List of images with perceptual hashes

        Returns:
            List of similar image groups
        """
        from .fast_search import FastSimilaritySearcher

        logger.info("Using FAISS for fast similarity search")

        # For now, use primary hash method (first in list)
        primary_method = self.hash_methods[0]

        # Build hash dictionary for FAISS
        hashes = {}
        image_map = {}  # Map image ID to image record

        for img in hashed_images:
            if (
                primary_method == HashMethod.DHASH
                and img.metadata.perceptual_hash_dhash
            ):
                hashes[img.id] = img.metadata.perceptual_hash_dhash
                image_map[img.id] = img
            elif (
                primary_method == HashMethod.AHASH
                and img.metadata.perceptual_hash_ahash
            ):
                hashes[img.id] = img.metadata.perceptual_hash_ahash
                image_map[img.id] = img
            elif (
                primary_method == HashMethod.WHASH
                and img.metadata.perceptual_hash_whash
            ):
                hashes[img.id] = img.metadata.perceptual_hash_whash
                image_map[img.id] = img

        if not hashes:
            logger.warning("No valid hashes found for FAISS search")
            return []

        # Build FAISS index
        searcher = FastSimilaritySearcher(
            hash_size=64, use_gpu=self.use_gpu  # 64-bit hash
        )
        searcher.build_index(hashes, method=primary_method.value)

        # Find similar pairs
        similar_pairs = searcher.find_similar(
            threshold=self.similarity_threshold,
            k=100,  # Search up to 100 nearest neighbors
        )

        logger.info(f"FAISS found {len(similar_pairs)} similar pairs")

        # Group similar images into clusters
        # Use union-find to merge overlapping pairs
        from collections import defaultdict

        neighbors: Dict[str, Set[str]] = defaultdict(set)

        for id1, id2, dist in similar_pairs:
            neighbors[id1].add(id2)
            neighbors[id2].add(id1)

        # Build groups from connected components
        visited: Set[str] = set()
        groups: List[DuplicateGroup] = []

        for image_id in neighbors.keys():
            if image_id in visited:
                continue

            # BFS to find all connected images
            cluster = {image_id}
            queue = [image_id]
            visited.add(image_id)

            while queue:
                current = queue.pop(0)
                for neighbor in neighbors[current]:
                    if neighbor not in visited:
                        cluster.add(neighbor)
                        queue.append(neighbor)
                        visited.add(neighbor)

            # Create group if cluster has multiple images
            if len(cluster) > 1:
                cluster_list = list(cluster)

                # Compute similarity metrics for all pairs in this group
                similarity_metrics = {}
                for i, id1 in enumerate(cluster_list):
                    for id2 in cluster_list[i + 1 :]:
                        image1 = image_map[id1]
                        image2 = image_map[id2]
                        metrics = self._compute_similarity_metrics(image1, image2)
                        if metrics:
                            key = self._make_pair_key(id1, id2)
                            similarity_metrics[key] = metrics

                # Use first image's hash as group hash
                first_img = image_map[cluster_list[0]]
                group_hash = None
                if primary_method == HashMethod.DHASH:
                    group_hash = first_img.metadata.perceptual_hash_dhash
                elif primary_method == HashMethod.AHASH:
                    group_hash = first_img.metadata.perceptual_hash_ahash
                elif primary_method == HashMethod.WHASH:
                    group_hash = first_img.metadata.perceptual_hash_whash

                group = DuplicateGroup(
                    id=f"similar_{cluster_list[0][:16]}",
                    images=cluster_list,
                    perceptual_hash=group_hash,
                    similarity_metrics=similarity_metrics,
                )
                groups.append(group)

        logger.info(f"Created {len(groups)} groups from FAISS results")
        return groups

    @staticmethod
    def _make_pair_key(id1: str, id2: str) -> str:
        """
        Create a consistent key for an image pair.

        Args:
            id1: First image ID
            id2: Second image ID

        Returns:
            Consistent key format (sorted IDs separated by colon)
        """
        # Sort IDs to ensure consistent key regardless of order
        sorted_ids = sorted([id1, id2])
        return f"{sorted_ids[0]}:{sorted_ids[1]}"

    def _compute_similarity_metrics(
        self, image1: ImageRecord, image2: ImageRecord
    ) -> Optional[SimilarityMetrics]:
        """
        Compute detailed similarity metrics between two images.

        Computes hamming distance and similarity percentage for all configured hash methods.

        Args:
            image1: First image
            image2: Second image

        Returns:
            SimilarityMetrics object with detailed comparison, or None if hashes are missing
        """
        metrics = SimilarityMetrics()

        try:
            # Compute metrics for each configured hash method
            distances = []
            similarities = []

            for method in self.hash_methods:
                hash1 = None
                hash2 = None

                if method == HashMethod.DHASH:
                    hash1 = image1.metadata.perceptual_hash_dhash
                    hash2 = image2.metadata.perceptual_hash_dhash
                    if hash1 and hash2:
                        metrics.dhash_distance = hamming_distance(hash1, hash2)
                        metrics.dhash_similarity = similarity_score(
                            hash1, hash2, self.hash_size
                        )
                        distances.append(metrics.dhash_distance)
                        similarities.append(metrics.dhash_similarity)
                elif method == HashMethod.AHASH:
                    hash1 = image1.metadata.perceptual_hash_ahash
                    hash2 = image2.metadata.perceptual_hash_ahash
                    if hash1 and hash2:
                        metrics.ahash_distance = hamming_distance(hash1, hash2)
                        metrics.ahash_similarity = similarity_score(
                            hash1, hash2, self.hash_size
                        )
                        distances.append(metrics.ahash_distance)
                        similarities.append(metrics.ahash_similarity)
                elif method == HashMethod.WHASH:
                    hash1 = image1.metadata.perceptual_hash_whash
                    hash2 = image2.metadata.perceptual_hash_whash
                    if hash1 and hash2:
                        metrics.whash_distance = hamming_distance(hash1, hash2)
                        metrics.whash_similarity = similarity_score(
                            hash1, hash2, self.hash_size
                        )
                        distances.append(metrics.whash_distance)
                        similarities.append(metrics.whash_similarity)

            # Compute overall similarity as average of all methods
            if similarities:
                metrics.overall_similarity = sum(similarities) / len(similarities)

            return metrics

        except Exception as e:
            logger.error(f"Error computing similarity metrics: {e}")
            return None

    def _are_images_similar(self, image1: ImageRecord, image2: ImageRecord) -> bool:
        """
        Check if two images are similar based on perceptual hashes.

        Uses all configured hash methods for better accuracy.
        All hash methods must agree (within threshold) for images to be considered similar.

        Args:
            image1: First image
            image2: Second image

        Returns:
            True if images are similar
        """
        try:
            # Check all configured hash methods
            for method in self.hash_methods:
                hash1 = None
                hash2 = None

                if method == HashMethod.DHASH:
                    hash1 = image1.metadata.perceptual_hash_dhash
                    hash2 = image2.metadata.perceptual_hash_dhash
                elif method == HashMethod.AHASH:
                    hash1 = image1.metadata.perceptual_hash_ahash
                    hash2 = image2.metadata.perceptual_hash_ahash
                elif method == HashMethod.WHASH:
                    hash1 = image1.metadata.perceptual_hash_whash
                    hash2 = image2.metadata.perceptual_hash_whash

                # If either hash is missing, not similar
                if not hash1 or not hash2:
                    return False

                # Check if this hash method shows similarity
                distance = hamming_distance(hash1, hash2)
                if distance > self.similarity_threshold:
                    return False

            # All hash methods agree - images are similar
            return True

        except Exception as e:
            logger.error(f"Error comparing hashes: {e}")
            return False

    def _merge_duplicate_groups(
        self, exact_groups: List[DuplicateGroup], similar_groups: List[DuplicateGroup]
    ) -> List[DuplicateGroup]:
        """
        Merge overlapping duplicate groups.

        If an image appears in both an exact and similar group,
        merge them into a single group.

        Args:
            exact_groups: Groups of exact duplicates
            similar_groups: Groups of similar images

        Returns:
            Merged list of duplicate groups
        """
        # Track which images are in which groups
        image_to_groups: Dict[str, List[int]] = defaultdict(list)

        all_groups = exact_groups + similar_groups

        for idx, group in enumerate(all_groups):
            for image_id in group.images:
                image_to_groups[image_id].append(idx)

        # Find groups that need merging
        merged_indices: Set[int] = set()
        final_groups: List[DuplicateGroup] = []

        for idx, group in enumerate(all_groups):
            if idx in merged_indices:
                continue

            # Find all groups that share images with this one
            related_indices = set([idx])
            to_check = set([idx])

            while to_check:
                check_idx = to_check.pop()
                check_group = all_groups[check_idx]

                for image_id in check_group.images:
                    for related_idx in image_to_groups[image_id]:
                        if related_idx not in related_indices:
                            related_indices.add(related_idx)
                            to_check.add(related_idx)

            # Merge all related groups
            merged_images = set()
            merged_metrics = {}
            for related_idx in related_indices:
                merged_images.update(all_groups[related_idx].images)
                # Merge similarity metrics from all related groups
                merged_metrics.update(all_groups[related_idx].similarity_metrics)
                merged_indices.add(related_idx)

            # Create merged group
            merged_group = DuplicateGroup(
                id=f"merged_{group.id}",
                images=list(merged_images),
                perceptual_hash=group.perceptual_hash,
                similarity_metrics=merged_metrics,
            )
            final_groups.append(merged_group)

        return final_groups

    def _score_and_select_primary(self, groups: List[DuplicateGroup]) -> None:
        """
        Score quality for each image and select primary in each group.

        Args:
            groups: List of duplicate groups to process
        """
        logger.info("Scoring image quality and selecting primary copies")

        for group in groups:
            # Get metadata for all images in group
            images_data = {}
            for image_id in group.images:
                image = self.catalog.get_image(image_id)
                if image:
                    images_data[image_id] = (image.metadata, image.file_type)

            if not images_data:
                logger.warning(f"No images found for group {group.id}")
                continue

            # Calculate quality scores for all images
            for image_id, (metadata, file_type) in images_data.items():
                score = calculate_quality_score(metadata, file_type)
                group.quality_scores[image_id] = score

            # Select best image as primary
            try:
                primary_id, _ = select_best(images_data)
                group.primary = primary_id
                logger.debug(f"Selected {primary_id} as primary for group {group.id}")
            except Exception as e:
                logger.error(f"Error selecting primary for group {group.id}: {e}")
                # Default to first image
                group.primary = group.images[0]

            # Check for date conflicts
            dates = set()
            for image_id in group.images:
                image = self.catalog.get_image(image_id)
                if image and image.dates.selected_date:
                    dates.add(image.dates.selected_date.date())

            if len(dates) > 1:
                group.date_conflict = True
                group.needs_review = True
                logger.info(
                    f"Date conflict in group {group.id}: {len(dates)} different dates"
                )

    def save_duplicate_groups(self) -> None:
        """Save detected duplicate groups to catalog."""
        logger.info(f"Saving {len(self.duplicate_groups)} duplicate groups to catalog")
        self.catalog.save_duplicate_groups(self.duplicate_groups)
        logger.info("Duplicate groups saved")

    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about detected duplicates.

        Returns:
            Dict with statistics
        """
        total_groups = len(self.duplicate_groups)
        total_duplicates = sum(len(g.images) for g in self.duplicate_groups)
        total_unique = total_groups  # One primary per group
        total_redundant = total_duplicates - total_unique
        groups_with_conflicts = sum(1 for g in self.duplicate_groups if g.needs_review)

        return {
            "total_groups": total_groups,
            "total_images_in_groups": total_duplicates,
            "total_unique": total_unique,
            "total_redundant": total_redundant,
            "groups_needing_review": groups_with_conflicts,
        }
