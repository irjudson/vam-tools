"""
Scan statistics tracking for job results.

Provides detailed breakdown of what happened during a scan job,
including why files were skipped or failed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class ScanStatistics:
    """
    Comprehensive statistics for a scan/analyze job.

    Tracks all files discovered and categorizes them by outcome
    to help diagnose discrepancies between files scanned and files in catalog.
    """

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    # File discovery
    files_discovered: int = 0  # Total files found in filesystem
    directories_scanned: int = 0

    # File processing outcomes
    files_added: int = 0  # Successfully added to catalog
    files_updated: int = 0  # Updated existing catalog entry

    # Skip reasons (mutually exclusive)
    skipped_already_in_catalog: int = 0  # Exact match already exists
    skipped_duplicate_checksum: int = 0  # Same checksum as another file in batch
    skipped_hidden_file: int = 0  # Starts with . or is in hidden directory
    skipped_synology_metadata: int = 0  # @eaDir, @SynoResource, etc.
    skipped_unsupported_format: int = 0  # Not an image or video
    skipped_file_not_accessible: int = 0  # Permission denied, doesn't exist

    # Error reasons
    errors_metadata_extraction: int = 0  # Failed to extract EXIF/metadata
    errors_checksum_computation: int = 0  # Failed to compute file hash
    errors_thumbnail_generation: int = 0  # Failed to create thumbnail
    errors_database: int = 0  # Failed to insert into database
    errors_other: int = 0  # Other errors

    # File type breakdown
    images_processed: int = 0
    videos_processed: int = 0

    # Size statistics
    total_bytes_processed: int = 0
    largest_file_bytes: int = 0
    largest_file_path: str = ""

    # Thumbnail stats
    thumbnails_generated: int = 0
    thumbnails_skipped_existing: int = 0
    thumbnails_failed: int = 0

    # Error details (sample of errors for debugging)
    error_samples: List[Dict[str, str]] = field(default_factory=list)
    max_error_samples: int = 50

    def record_error(self, file_path: str, error_type: str, error_msg: str) -> None:
        """Record an error with file path for debugging."""
        if len(self.error_samples) < self.max_error_samples:
            self.error_samples.append(
                {
                    "file": str(file_path),
                    "type": error_type,
                    "message": str(error_msg)[:200],  # Truncate long messages
                }
            )

    @property
    def duration_seconds(self) -> float:
        """Get scan duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def files_per_second(self) -> float:
        """Calculate processing throughput."""
        duration = self.duration_seconds
        if duration > 0:
            return self.total_files_processed / duration
        return 0.0

    @property
    def total_files_processed(self) -> int:
        """Total files that were actually processed (success or failure)."""
        return (
            self.files_added
            + self.files_updated
            + self.total_skipped
            + self.total_errors
        )

    @property
    def total_skipped(self) -> int:
        """Total files skipped for any reason."""
        return (
            self.skipped_already_in_catalog
            + self.skipped_duplicate_checksum
            + self.skipped_hidden_file
            + self.skipped_synology_metadata
            + self.skipped_unsupported_format
            + self.skipped_file_not_accessible
        )

    @property
    def total_errors(self) -> int:
        """Total files that had errors."""
        return (
            self.errors_metadata_extraction
            + self.errors_checksum_computation
            + self.errors_thumbnail_generation
            + self.errors_database
            + self.errors_other
        )

    def finish(self) -> None:
        """Mark the scan as complete."""
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            # Summary
            "status": "completed",
            "duration_seconds": round(self.duration_seconds, 2),
            "files_per_second": round(self.files_per_second, 2),
            "started_at": datetime.fromtimestamp(self.start_time).isoformat(),
            "completed_at": (
                datetime.fromtimestamp(self.end_time).isoformat()
                if self.end_time > 0
                else None
            ),
            # Counts
            "files_discovered": self.files_discovered,
            "directories_scanned": self.directories_scanned,
            "total_processed": self.total_files_processed,
            # Outcomes
            "files_added": self.files_added,
            "files_updated": self.files_updated,
            "total_skipped": self.total_skipped,
            "total_errors": self.total_errors,
            # Skip breakdown
            "skip_reasons": {
                "already_in_catalog": self.skipped_already_in_catalog,
                "duplicate_checksum": self.skipped_duplicate_checksum,
                "hidden_file": self.skipped_hidden_file,
                "synology_metadata": self.skipped_synology_metadata,
                "unsupported_format": self.skipped_unsupported_format,
                "not_accessible": self.skipped_file_not_accessible,
            },
            # Error breakdown
            "error_reasons": {
                "metadata_extraction": self.errors_metadata_extraction,
                "checksum_computation": self.errors_checksum_computation,
                "thumbnail_generation": self.errors_thumbnail_generation,
                "database": self.errors_database,
                "other": self.errors_other,
            },
            # File types
            "file_types": {
                "images": self.images_processed,
                "videos": self.videos_processed,
            },
            # Size stats
            "size_stats": {
                "total_bytes": self.total_bytes_processed,
                "total_gb": round(self.total_bytes_processed / (1024**3), 2),
                "largest_file_bytes": self.largest_file_bytes,
                "largest_file": self.largest_file_path,
            },
            # Thumbnails
            "thumbnails": {
                "generated": self.thumbnails_generated,
                "skipped_existing": self.thumbnails_skipped_existing,
                "failed": self.thumbnails_failed,
            },
            # Error samples for debugging
            "error_samples": self.error_samples,
        }

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Scan completed in {self.duration_seconds:.1f}s ({self.files_per_second:.1f} files/sec)",
            "",
            f"Files discovered: {self.files_discovered:,}",
            f"Files added to catalog: {self.files_added:,}",
            f"Files updated: {self.files_updated:,}",
            "",
            f"Skipped ({self.total_skipped:,} total):",
        ]

        if self.skipped_already_in_catalog:
            lines.append(f"  - Already in catalog: {self.skipped_already_in_catalog:,}")
        if self.skipped_duplicate_checksum:
            lines.append(f"  - Duplicate checksum: {self.skipped_duplicate_checksum:,}")
        if self.skipped_hidden_file:
            lines.append(f"  - Hidden files: {self.skipped_hidden_file:,}")
        if self.skipped_synology_metadata:
            lines.append(f"  - Synology metadata: {self.skipped_synology_metadata:,}")
        if self.skipped_unsupported_format:
            lines.append(f"  - Unsupported format: {self.skipped_unsupported_format:,}")
        if self.skipped_file_not_accessible:
            lines.append(f"  - Not accessible: {self.skipped_file_not_accessible:,}")

        if self.total_errors:
            lines.append("")
            lines.append(f"Errors ({self.total_errors:,} total):")
            if self.errors_metadata_extraction:
                lines.append(
                    f"  - Metadata extraction: {self.errors_metadata_extraction:,}"
                )
            if self.errors_checksum_computation:
                lines.append(
                    f"  - Checksum computation: {self.errors_checksum_computation:,}"
                )
            if self.errors_thumbnail_generation:
                lines.append(
                    f"  - Thumbnail generation: {self.errors_thumbnail_generation:,}"
                )
            if self.errors_database:
                lines.append(f"  - Database errors: {self.errors_database:,}")
            if self.errors_other:
                lines.append(f"  - Other errors: {self.errors_other:,}")

        lines.append("")
        lines.append(
            f"File types: {self.images_processed:,} images, {self.videos_processed:,} videos"
        )
        lines.append(f"Total size: {self.total_bytes_processed / (1024**3):.2f} GB")

        return "\n".join(lines)
