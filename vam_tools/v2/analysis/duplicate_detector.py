"""
Duplicate detection system.

Identifies duplicate and similar images using:
1. Exact duplicates (same checksum)
2. Near duplicates (similar perceptual hash)
3. Quality scoring to select the best copy
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from ..core.catalog import CatalogDatabase
from ..core.types import DuplicateGroup, ImageRecord
from .perceptual_hash import combined_hash, hamming_distance
from .quality_scorer import calculate_quality_score, select_best

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """Detect and manage duplicate images in a catalog."""

    def __init__(
        self,
        catalog: CatalogDatabase,
        similarity_threshold: int = 5,
        hash_size: int = 8,
    ):
        """
        Initialize duplicate detector.

        Args:
            catalog: Catalog database to analyze
            similarity_threshold: Maximum Hamming distance for similar images (default: 5)
            hash_size: Size of perceptual hash (default: 8 = 64-bit)
        """
        self.catalog = catalog
        self.similarity_threshold = similarity_threshold
        self.hash_size = hash_size
        self.duplicate_groups: List[DuplicateGroup] = []

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

        # Step 1: Compute perceptual hashes
        self._compute_perceptual_hashes(recompute_hashes)

        # Step 2: Find exact duplicates (same checksum)
        exact_groups = self._find_exact_duplicates()
        logger.info(f"Found {len(exact_groups)} exact duplicate groups")

        # Step 3: Find similar images (perceptual hash similarity)
        similar_groups = self._find_similar_images()
        logger.info(f"Found {len(similar_groups)} similar image groups")

        # Step 4: Merge and deduplicate groups
        all_groups = self._merge_duplicate_groups(exact_groups, similar_groups)
        logger.info(f"Total {len(all_groups)} duplicate groups after merging")

        # Step 5: Score quality and select primary for each group
        self._score_and_select_primary(all_groups)

        self.duplicate_groups = all_groups
        return all_groups

    def _compute_perceptual_hashes(self, force: bool = False) -> None:
        """
        Compute perceptual hashes for all images in catalog.

        Args:
            force: Force recomputation even if hash exists
        """
        images = self.catalog.get_all_images()
        images_to_process = []

        for image in images:
            # Check if we need to compute hash
            if force or not image.metadata.perceptual_hash_dhash:
                images_to_process.append(image)

        if not images_to_process:
            logger.info("All images already have perceptual hashes")
            return

        logger.info(f"Computing perceptual hashes for {len(images_to_process)} images")

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
                # Compute both hashes
                hashes = combined_hash(image.source_path, self.hash_size)

                if hashes:
                    dhash, ahash = hashes
                    # Update image metadata
                    image.metadata.perceptual_hash_dhash = dhash
                    image.metadata.perceptual_hash_ahash = ahash
                    # Save to catalog
                    self.catalog.update_image(image)
                else:
                    logger.warning(f"Failed to compute hash for {image.source_path}")

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

        for image in self.catalog.get_all_images():
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
        images = self.catalog.get_all_images()

        # Filter to images with valid perceptual hashes
        hashed_images = [
            img
            for img in images
            if img.metadata.perceptual_hash_dhash and img.metadata.perceptual_hash_ahash
        ]

        if not hashed_images:
            logger.warning("No images with perceptual hashes found")
            return []

        logger.info(
            f"Comparing {len(hashed_images)} images for similarity "
            f"(threshold: {self.similarity_threshold})"
        )

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

                for image2 in hashed_images[i + 1 :]:
                    if image2.id in grouped:
                        continue

                    # Compare perceptual hashes
                    if self._are_images_similar(image1, image2):
                        similar_ids.append(image2.id)
                        grouped.add(image2.id)

                # Create group if we found similar images
                if len(similar_ids) > 1:
                    group = DuplicateGroup(
                        id=f"similar_{image1.id[:16]}",
                        images=similar_ids,
                        perceptual_hash=image1.metadata.perceptual_hash_dhash,
                    )
                    groups.append(group)
                    grouped.add(image1.id)

                progress.advance(task)

        return groups

    def _are_images_similar(self, image1: ImageRecord, image2: ImageRecord) -> bool:
        """
        Check if two images are similar based on perceptual hashes.

        Uses both dHash and aHash for better accuracy.

        Args:
            image1: First image
            image2: Second image

        Returns:
            True if images are similar
        """
        # Both hashes must be similar
        dhash1 = image1.metadata.perceptual_hash_dhash
        dhash2 = image2.metadata.perceptual_hash_dhash
        ahash1 = image1.metadata.perceptual_hash_ahash
        ahash2 = image2.metadata.perceptual_hash_ahash

        if not all([dhash1, dhash2, ahash1, ahash2]):
            return False

        try:
            # Check dHash distance
            dhash_distance = hamming_distance(dhash1, dhash2)
            if dhash_distance > self.similarity_threshold:
                return False

            # Check aHash distance for confirmation
            ahash_distance = hamming_distance(ahash1, ahash2)
            if ahash_distance > self.similarity_threshold:
                return False

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
            for related_idx in related_indices:
                merged_images.update(all_groups[related_idx].images)
                merged_indices.add(related_idx)

            # Create merged group
            merged_group = DuplicateGroup(
                id=f"merged_{group.id}",
                images=list(merged_images),
                perceptual_hash=group.perceptual_hash,
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
