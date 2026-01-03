"""
Organization strategies for file management.

Defines directory structures, naming conventions, and organization rules.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from ..core.types import ImageRecord


class DirectoryStructure(str, Enum):
    """Directory structure patterns for organizing files."""

    YEAR_MONTH = "YYYY-MM"  # 2023-06
    YEAR_SLASH_MONTH = "YYYY/MM"  # 2023/06
    YEAR_MONTH_DAY = "YYYY-MM-DD"  # 2023-06-15
    YEAR_ONLY = "YYYY"  # 2023
    YEAR_SLASH_MONTH_DAY = "YYYY/MM-DD"  # 2023/06-15
    FLAT = "FLAT"  # All files in one directory


class NamingStrategy(str, Enum):
    """File naming strategies."""

    DATE_TIME_CHECKSUM = "date_time_checksum"  # 2023-06-15_143022_abc123.jpg
    DATE_TIME_ORIGINAL = "date_time_original"  # 2023-06-15_143022_IMG_1234.jpg
    ORIGINAL = "original"  # IMG_1234.jpg (keep original name)
    CHECKSUM = "checksum"  # abc123def456.jpg
    TIME_CHECKSUM = "time_checksum"  # 143022_abc12345.jpg


class OrganizationStrategy(BaseModel):
    """Strategy for organizing files."""

    directory_structure: DirectoryStructure = Field(
        default=DirectoryStructure.YEAR_MONTH,
        description="Directory structure pattern",
    )

    naming_strategy: NamingStrategy = Field(
        default=NamingStrategy.DATE_TIME_CHECKSUM,
        description="File naming strategy",
    )

    handle_duplicates: bool = Field(
        default=True,
        description="Handle duplicate files (add suffix if name conflicts)",
    )

    preserve_directory_structure: bool = Field(
        default=False,
        description="Preserve original directory structure under date directories",
    )

    model_config = ConfigDict(use_enum_values=True)

    def get_target_directory(
        self, base_path: Path, image: ImageRecord, use_mtime_fallback: bool = False
    ) -> Optional[Path]:
        """
        Get target directory for an image based on the strategy.

        Args:
            base_path: Base output directory
            image: Image record with metadata
            use_mtime_fallback: If True, fall back to file mtime when EXIF date missing

        Returns:
            Target directory path, or None if image has no date and fallback disabled
        """
        import os
        from datetime import datetime as dt

        # Try to get date from EXIF
        if image.dates and image.dates.selected_date:
            date = image.dates.selected_date
        elif use_mtime_fallback and image.source_path.exists():
            # Fall back to file modification time
            mtime = os.path.getmtime(image.source_path)
            date = dt.fromtimestamp(mtime)
        else:
            return None

        if self.directory_structure == DirectoryStructure.YEAR_MONTH:
            return base_path / date.strftime("%Y-%m")
        elif self.directory_structure == DirectoryStructure.YEAR_SLASH_MONTH:
            return base_path / date.strftime("%Y") / date.strftime("%m")
        elif self.directory_structure == DirectoryStructure.YEAR_MONTH_DAY:
            return base_path / date.strftime("%Y-%m-%d")
        elif self.directory_structure == DirectoryStructure.YEAR_ONLY:
            return base_path / date.strftime("%Y")
        elif self.directory_structure == DirectoryStructure.YEAR_SLASH_MONTH_DAY:
            return base_path / date.strftime("%Y") / date.strftime("%m-%d")
        elif self.directory_structure == DirectoryStructure.FLAT:
            return base_path

        # All enum cases covered
        return base_path  # type: ignore[unreachable]  # Defensive fallback

    def get_target_filename(self, image: ImageRecord) -> str:
        """
        Get target filename for an image based on the strategy.

        Args:
            image: Image record with metadata

        Returns:
            Target filename
        """
        original_name = image.source_path.name
        stem = image.source_path.stem
        suffix = image.source_path.suffix

        if self.naming_strategy == NamingStrategy.ORIGINAL:
            return original_name

        elif self.naming_strategy == NamingStrategy.CHECKSUM:
            return f"{image.checksum}{suffix}"

        elif self.naming_strategy == NamingStrategy.TIME_CHECKSUM:
            if image.dates and image.dates.selected_date:
                time_str = image.dates.selected_date.strftime("%H%M%S")
                checksum_short = image.checksum[:8]
                return f"{time_str}_{checksum_short}{suffix}"
            else:
                # Fall back to checksum if no date
                return f"{image.checksum}{suffix}"

        elif self.naming_strategy == NamingStrategy.DATE_TIME_CHECKSUM:
            if image.dates and image.dates.selected_date:
                date_str = image.dates.selected_date.strftime("%Y-%m-%d_%H%M%S")
                checksum_short = image.checksum[:8]
                return f"{date_str}_{checksum_short}{suffix}"
            else:
                # Fall back to checksum if no date
                return f"{image.checksum}{suffix}"

        elif self.naming_strategy == NamingStrategy.DATE_TIME_ORIGINAL:
            if image.dates and image.dates.selected_date:
                date_str = image.dates.selected_date.strftime("%Y-%m-%d_%H%M%S")
                return f"{date_str}_{stem}{suffix}"
            else:
                # Fall back to original if no date
                return original_name

        # All enum cases covered - this is for type checker
        return original_name  # type: ignore[unreachable]  # Defensive fallback

    def get_target_path(
        self, base_path: Path, image: ImageRecord, use_mtime_fallback: bool = False
    ) -> Optional[Path]:
        """
        Get complete target path for an image.

        Routes rejected images to _rejected/ subdirectory.

        Args:
            base_path: Base output directory
            image: Image record
            use_mtime_fallback: If True, fall back to file mtime when EXIF date missing

        Returns:
            Complete target path, or None if image has no date
        """
        # Adjust base path for rejected images
        if hasattr(image, "status_id") and image.status_id == "rejected":
            base_path = base_path / "_rejected"

        target_dir = self.get_target_directory(base_path, image, use_mtime_fallback)
        if not target_dir:
            return None

        filename = self.get_target_filename(image)
        return target_dir / filename

    def resolve_conflict_with_full_checksum(
        self, base_path: Path, image: ImageRecord, conflicting_path: Path
    ) -> Path:
        """
        Resolve naming conflict by using full checksum instead of short checksum.

        Args:
            base_path: Base output directory
            image: Image record
            conflicting_path: Path that already exists

        Returns:
            New path with full checksum
        """
        # Get directory and extension
        target_dir = conflicting_path.parent
        suffix = image.source_path.suffix

        # Generate filename with full checksum
        if image.dates and image.dates.selected_date:
            time_str = image.dates.selected_date.strftime("%H%M%S")
            return target_dir / f"{time_str}_{image.checksum}{suffix}"
        else:
            # No date - use checksum only
            return target_dir / f"{image.checksum}{suffix}"

    def resolve_naming_conflict(
        self, target_path: Path, image: ImageRecord
    ) -> Optional[Path]:
        """
        Resolve naming conflict by adding a suffix.

        Args:
            target_path: Original target path
            image: Image record

        Returns:
            Modified path with suffix to avoid conflict, or None if conflict handling is disabled
        """
        if not self.handle_duplicates:
            return None

        stem = target_path.stem
        suffix = target_path.suffix
        parent = target_path.parent

        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter:03d}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1
            # Safety check to avoid infinite loop
            if counter > 9999:
                raise ValueError(f"Too many naming conflicts for {target_path}")
