"""
Duplicate image detection using perceptual hashing.

This module provides functionality to detect duplicate images that may have:
- Different file sizes
- Different formats
- Different resolutions
- Different filenames

It uses multiple hashing algorithms:
- MD5 for exact file duplicates
- dHash (difference hash) for perceptual similarity
- aHash (average hash) for perceptual similarity
"""

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ImageHash:
    """Container for various hashes of an image."""

    file_hash: Optional[str]  # MD5 hash of file
    dhash: Optional[str]  # Difference hash
    ahash: Optional[str]  # Average hash


@dataclass
class DuplicateGroup:
    """A group of duplicate or similar images."""

    images: List[Path]
    similarity_type: str  # 'exact' or 'perceptual'
    hash_distance: int  # 0 for exact, hamming distance for perceptual


class DuplicateDetector:
    """Detect duplicate images using file and perceptual hashing."""

    def __init__(self, hash_size: int = 8) -> None:
        """
        Initialize the duplicate detector.

        Args:
            hash_size: Size of the perceptual hash (default 8 = 64 bits)
        """
        self.hash_size = hash_size
        self.image_hashes: Dict[Path, ImageHash] = {}

    def calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """
        Calculate MD5 hash of a file for exact duplicate detection.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal MD5 hash string, or None on error
        """
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.debug(f"Error calculating file hash for {file_path}: {e}")
            return None

    def calculate_dhash(self, image_path: Path) -> Optional[str]:
        """
        Calculate difference hash (dHash) for perceptual similarity.

        dHash works by comparing adjacent pixels in a resized image to create
        a fingerprint that is resistant to scaling and minor modifications.

        Args:
            image_path: Path to the image file

        Returns:
            Binary string representing the hash, or None on error
        """
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale and resize
                img = img.convert("L").resize(
                    (self.hash_size + 1, self.hash_size), Image.Resampling.LANCZOS
                )

                # Calculate horizontal gradient
                hash_bits = []
                for row in range(self.hash_size):
                    for col in range(self.hash_size):
                        pixel_left = img.getpixel((col, row))
                        pixel_right = img.getpixel((col + 1, row))
                        hash_bits.append("1" if pixel_left > pixel_right else "0")

                return "".join(hash_bits)
        except Exception as e:
            logger.debug(f"Error calculating dHash for {image_path}: {e}")
            return None

    def calculate_ahash(self, image_path: Path) -> Optional[str]:
        """
        Calculate average hash (aHash) for perceptual similarity.

        aHash works by comparing each pixel to the average pixel value in
        a resized image.

        Args:
            image_path: Path to the image file

        Returns:
            Binary string representing the hash, or None on error
        """
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale and resize
                img = img.convert("L").resize(
                    (self.hash_size, self.hash_size), Image.Resampling.LANCZOS
                )

                # Calculate average pixel value
                pixels = list(img.getdata())
                avg = sum(pixels) / len(pixels)

                # Generate hash based on average
                hash_bits = ["1" if pixel > avg else "0" for pixel in pixels]
                return "".join(hash_bits)
        except Exception as e:
            logger.debug(f"Error calculating aHash for {image_path}: {e}")
            return None

    def calculate_hashes(self, image_path: Path) -> ImageHash:
        """
        Calculate all hashes for an image.

        Args:
            image_path: Path to the image file

        Returns:
            ImageHash object containing all calculated hashes
        """
        return ImageHash(
            file_hash=self.calculate_file_hash(image_path),
            dhash=self.calculate_dhash(image_path),
            ahash=self.calculate_ahash(image_path),
        )

    @staticmethod
    def hamming_distance(hash1: Optional[str], hash2: Optional[str]) -> int:
        """
        Calculate Hamming distance between two hash strings.

        The Hamming distance is the number of positions at which the
        corresponding bits are different.

        Args:
            hash1: First hash string
            hash2: Second hash string

        Returns:
            Hamming distance (0 = identical, higher = more different)
            Returns a very large number if hashes are incompatible
        """
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 999999

        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def process_images(self, image_paths: List[Path]) -> None:
        """
        Calculate hashes for all images.

        Args:
            image_paths: List of paths to image files
        """
        self.image_hashes = {}

        for image_path in image_paths:
            try:
                hashes = self.calculate_hashes(image_path)
                self.image_hashes[image_path] = hashes
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")

    def find_exact_duplicates(self) -> List[DuplicateGroup]:
        """
        Find exact file duplicates based on MD5 hash.

        Returns:
            List of DuplicateGroup objects containing exact duplicates
        """
        file_hash_groups: Dict[str, List[Path]] = defaultdict(list)

        # Group by file hash
        for image_path, hashes in self.image_hashes.items():
            if hashes.file_hash:
                file_hash_groups[hashes.file_hash].append(image_path)

        # Create duplicate groups
        duplicate_groups = []
        for file_hash, images in file_hash_groups.items():
            if len(images) > 1:
                duplicate_groups.append(
                    DuplicateGroup(
                        images=images,
                        similarity_type="exact",
                        hash_distance=0,
                    )
                )

        return duplicate_groups

    def find_perceptual_duplicates(
        self, threshold: int = 5
    ) -> List[DuplicateGroup]:
        """
        Find perceptually similar images using perceptual hashing.

        Args:
            threshold: Maximum Hamming distance to consider images similar
                      (0-64 for hash_size=8, lower = more strict)

        Returns:
            List of DuplicateGroup objects containing similar images
        """
        # First, find exact duplicates to exclude them
        exact_duplicate_images: Set[Path] = set()
        for group in self.find_exact_duplicates():
            exact_duplicate_images.update(group.images)

        # Get images not in exact duplicate groups
        remaining_images = [
            img for img in self.image_hashes.keys()
            if img not in exact_duplicate_images
        ]

        duplicate_groups = []
        processed: Set[Path] = set()

        for i, image1 in enumerate(remaining_images):
            if image1 in processed:
                continue

            current_group = [image1]
            processed.add(image1)

            hashes1 = self.image_hashes[image1]

            for image2 in remaining_images[i + 1 :]:
                if image2 in processed:
                    continue

                hashes2 = self.image_hashes[image2]

                # Check both dHash and aHash
                dhash_dist = self.hamming_distance(hashes1.dhash, hashes2.dhash)
                ahash_dist = self.hamming_distance(hashes1.ahash, hashes2.ahash)

                # Consider similar if either hash is within threshold
                min_distance = min(dhash_dist, ahash_dist)
                if min_distance <= threshold:
                    current_group.append(image2)
                    processed.add(image2)

            if len(current_group) > 1:
                # Calculate average distance for the group
                distances = []
                for j in range(len(current_group)):
                    for k in range(j + 1, len(current_group)):
                        h1 = self.image_hashes[current_group[j]]
                        h2 = self.image_hashes[current_group[k]]
                        d = min(
                            self.hamming_distance(h1.dhash, h2.dhash),
                            self.hamming_distance(h1.ahash, h2.ahash),
                        )
                        distances.append(d)

                avg_distance = (
                    int(sum(distances) / len(distances)) if distances else 0
                )

                duplicate_groups.append(
                    DuplicateGroup(
                        images=current_group,
                        similarity_type="perceptual",
                        hash_distance=avg_distance,
                    )
                )

        return duplicate_groups

    def find_all_duplicates(
        self, image_paths: List[Path], threshold: int = 5
    ) -> List[DuplicateGroup]:
        """
        Find both exact and perceptual duplicates.

        Args:
            image_paths: List of paths to image files
            threshold: Maximum Hamming distance for perceptual duplicates

        Returns:
            List of all duplicate groups (exact and perceptual)
        """
        self.process_images(image_paths)

        exact_duplicates = self.find_exact_duplicates()
        perceptual_duplicates = self.find_perceptual_duplicates(threshold)

        return exact_duplicates + perceptual_duplicates
