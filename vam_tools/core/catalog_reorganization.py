"""
Lightroom catalog reorganization.

This module provides functionality to reorganize photo files into a structured
directory hierarchy based on their dates. Supports dry-run mode for safe testing.
"""

import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from .date_extraction import DateExtractor, DateInfo

logger = logging.getLogger(__name__)


class ConflictResolution(Enum):
    """Strategy for handling filename conflicts."""

    SKIP = "skip"  # Skip the file if destination exists
    RENAME = "rename"  # Rename with counter (e.g., file_1.jpg)
    OVERWRITE = "overwrite"  # Overwrite existing file


class OrganizationStrategy(Enum):
    """Directory organization strategy."""

    YEAR_MONTH_DAY = "year/month-day"  # 2023/12-25/
    YEAR_MONTH = "year/month"  # 2023/12/
    YEAR = "year"  # 2023/
    FLAT_DATE = "flat"  # 2023-12-25/


@dataclass
class ReorganizationPlan:
    """Plan for reorganizing a single file."""

    source: Path
    destination: Path
    date_info: Optional[DateInfo]
    action: str  # 'move', 'copy', or 'skip'
    reason: str  # Human-readable reason for the action


class CatalogReorganizer:
    """Reorganize photo catalog into date-based directory structure."""

    def __init__(
        self,
        output_directory: Path,
        strategy: OrganizationStrategy = OrganizationStrategy.YEAR_MONTH_DAY,
        conflict_resolution: ConflictResolution = ConflictResolution.RENAME,
        copy_mode: bool = False,
    ) -> None:
        """
        Initialize the catalog reorganizer.

        Args:
            output_directory: Root directory for organized photos
            strategy: Directory organization strategy
            conflict_resolution: How to handle filename conflicts
            copy_mode: If True, copy files instead of moving them
        """
        self.output_directory = output_directory
        self.strategy = strategy
        self.conflict_resolution = conflict_resolution
        self.copy_mode = copy_mode
        self.no_date_counter: Dict[str, int] = defaultdict(int)

    def generate_destination_path(
        self, source_path: Path, date_info: Optional[DateInfo]
    ) -> Path:
        """
        Generate destination path for a file based on its date.

        Args:
            source_path: Original file path
            date_info: Date information for the file

        Returns:
            Destination path for the file
        """
        if date_info is None:
            # Files without dates go to "unknown" directory with counter
            ext = source_path.suffix
            counter = self.no_date_counter[ext]
            self.no_date_counter[ext] += 1
            return self.output_directory / "unknown" / f"nodate_{counter}{ext}"

        date = date_info.date

        # Build directory path based on strategy
        if self.strategy == OrganizationStrategy.YEAR_MONTH_DAY:
            dir_path = (
                self.output_directory
                / str(date.year)
                / f"{date.month:02d}-{date.day:02d}"
            )
        elif self.strategy == OrganizationStrategy.YEAR_MONTH:
            dir_path = self.output_directory / str(date.year) / f"{date.month:02d}"
        elif self.strategy == OrganizationStrategy.YEAR:
            dir_path = self.output_directory / str(date.year)
        else:  # FLAT_DATE
            dir_path = (
                self.output_directory / f"{date.year}-{date.month:02d}-{date.day:02d}"
            )

        # Generate filename based on date and original filename
        timestamp = date.format("YYYY-MM-DD_HHmmss")
        original_name = source_path.stem
        ext = source_path.suffix

        filename = f"{timestamp}_{original_name}{ext}"
        return dir_path / filename

    def resolve_conflict(self, destination: Path) -> Path:
        """
        Resolve filename conflict based on strategy.

        Args:
            destination: Proposed destination path

        Returns:
            Final destination path (may be modified to avoid conflicts)
        """
        if not destination.exists():
            return destination

        if self.conflict_resolution == ConflictResolution.SKIP:
            return destination

        if self.conflict_resolution == ConflictResolution.OVERWRITE:
            return destination

        # RENAME strategy: add counter
        counter = 1
        stem = destination.stem
        ext = destination.suffix
        parent = destination.parent

        while destination.exists():
            new_name = f"{stem}_{counter}{ext}"
            destination = parent / new_name
            counter += 1

        return destination

    def create_reorganization_plan(
        self, image_paths: List[Path]
    ) -> List[ReorganizationPlan]:
        """
        Create a plan for reorganizing files.

        Args:
            image_paths: List of image file paths to reorganize

        Returns:
            List of ReorganizationPlan objects
        """
        plans: List[ReorganizationPlan] = []

        with DateExtractor() as extractor:
            for image_path in image_paths:
                try:
                    # Extract date information
                    date_info = extractor.extract_earliest_date(image_path)

                    # Generate destination path
                    destination = self.generate_destination_path(image_path, date_info)

                    # Check for conflicts
                    if destination.exists():
                        if self.conflict_resolution == ConflictResolution.SKIP:
                            plans.append(
                                ReorganizationPlan(
                                    source=image_path,
                                    destination=destination,
                                    date_info=date_info,
                                    action="skip",
                                    reason="File already exists at destination",
                                )
                            )
                            continue
                        else:
                            destination = self.resolve_conflict(destination)

                    # Determine action
                    action = "copy" if self.copy_mode else "move"
                    reason = (
                        f"Date from {date_info.source}"
                        if date_info
                        else "No date found"
                    )

                    plans.append(
                        ReorganizationPlan(
                            source=image_path,
                            destination=destination,
                            date_info=date_info,
                            action=action,
                            reason=reason,
                        )
                    )

                except Exception as e:
                    logger.error(f"Error creating plan for {image_path}: {e}")
                    plans.append(
                        ReorganizationPlan(
                            source=image_path,
                            destination=image_path,
                            date_info=None,
                            action="skip",
                            reason=f"Error: {e}",
                        )
                    )

        return plans

    def execute_plan(
        self, plans: List[ReorganizationPlan], dry_run: bool = False
    ) -> Dict[str, int]:
        """
        Execute the reorganization plan.

        Args:
            plans: List of ReorganizationPlan objects
            dry_run: If True, only simulate the reorganization

        Returns:
            Dictionary with statistics:
            - moved: Number of files moved
            - copied: Number of files copied
            - skipped: Number of files skipped
            - errors: Number of errors
        """
        stats = {"moved": 0, "copied": 0, "skipped": 0, "errors": 0}

        for plan in plans:
            try:
                if plan.action == "skip":
                    logger.info(f"Skipping {plan.source}: {plan.reason}")
                    stats["skipped"] += 1
                    continue

                if dry_run:
                    logger.info(
                        f"[DRY RUN] Would {plan.action} {plan.source} -> {plan.destination}"
                    )
                    logger.info(f"[DRY RUN] Reason: {plan.reason}")
                    stats[f"{plan.action}d"] = stats.get(f"{plan.action}d", 0) + 1
                else:
                    # Create destination directory
                    plan.destination.parent.mkdir(parents=True, exist_ok=True)

                    if plan.action == "move":
                        shutil.move(str(plan.source), str(plan.destination))
                        logger.info(f"Moved {plan.source} -> {plan.destination}")
                        stats["moved"] += 1
                    elif plan.action == "copy":
                        shutil.copy2(str(plan.source), str(plan.destination))
                        logger.info(f"Copied {plan.source} -> {plan.destination}")
                        stats["copied"] += 1

            except Exception as e:
                logger.error(f"Error executing plan for {plan.source}: {e}")
                stats["errors"] += 1

        return stats

    def reorganize(
        self, image_paths: List[Path], dry_run: bool = False
    ) -> Dict[str, int]:
        """
        Reorganize images into date-based directory structure.

        Args:
            image_paths: List of image file paths to reorganize
            dry_run: If True, only simulate the reorganization

        Returns:
            Dictionary with statistics about the reorganization
        """
        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Starting reorganization of "
            f"{len(image_paths)} images"
        )
        logger.info(f"Output directory: {self.output_directory}")
        logger.info(f"Strategy: {self.strategy.value}")
        logger.info(f"Mode: {'copy' if self.copy_mode else 'move'}")

        plans = self.create_reorganization_plan(image_paths)
        stats = self.execute_plan(plans, dry_run=dry_run)

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Reorganization complete. "
            f"Stats: {stats}"
        )

        return stats
