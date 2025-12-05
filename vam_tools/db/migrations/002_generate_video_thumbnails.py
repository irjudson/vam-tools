#!/usr/bin/env python3
"""
Migration 002: Generate thumbnails for videos.

Videos were previously not getting thumbnails during scan. This script
generates thumbnails for all videos that don't have them.

Run with: python -m vam_tools.db.migrations.002_generate_video_thumbnails
"""

import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sqlalchemy import text  # noqa: E402

from vam_tools.db.connection import engine  # noqa: E402
from vam_tools.shared.thumbnail_utils import (  # noqa: E402
    generate_thumbnail,
    get_thumbnail_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default catalog path for thumbnails
DEFAULT_CATALOG_PATH = Path("/app/catalogs/default")


def run_migration(
    catalog_path: Path = DEFAULT_CATALOG_PATH, dry_run: bool = False
) -> dict:
    """
    Generate thumbnails for videos that don't have them.

    Args:
        catalog_path: Path to catalog directory (for storing thumbnails)
        dry_run: If True, only report what would be changed without generating.

    Returns:
        Dictionary with migration statistics.
    """
    stats = {
        "total_videos": 0,
        "videos_with_thumbnails": 0,
        "videos_without_thumbnails": 0,
        "thumbnails_generated": 0,
        "errors": 0,
    }

    thumbnails_dir = catalog_path / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    with engine.connect() as conn:
        # Get video counts
        result = conn.execute(
            text(
                """
            SELECT
                COUNT(*) as total,
                COUNT(thumbnail_path) as with_thumb
            FROM images
            WHERE file_type = 'video'
        """
            )
        )
        row = result.fetchone()
        stats["total_videos"] = row[0]
        stats["videos_with_thumbnails"] = row[1]
        stats["videos_without_thumbnails"] = row[0] - row[1]

        logger.info(f"Total videos: {stats['total_videos']:,}")
        logger.info(f"Videos with thumbnails: {stats['videos_with_thumbnails']:,}")
        logger.info(
            f"Videos without thumbnails: {stats['videos_without_thumbnails']:,}"
        )

        if dry_run:
            logger.info("DRY RUN - no thumbnails will be generated")
            return stats

        if stats["videos_without_thumbnails"] == 0:
            logger.info("All videos already have thumbnails!")
            return stats

        # Get videos without thumbnails
        result = conn.execute(
            text(
                """
            SELECT id, source_path, checksum
            FROM images
            WHERE file_type = 'video'
              AND thumbnail_path IS NULL
            ORDER BY created_at DESC
        """
            )
        )
        videos = result.fetchall()

        logger.info(f"Generating thumbnails for {len(videos)} videos...")

        for i, video in enumerate(videos):
            video_id = video[0]
            source_path = Path(video[1])
            checksum = video[2]

            if (i + 1) % 50 == 0:
                logger.info(
                    f"Progress: {i + 1}/{len(videos)} "
                    f"({stats['thumbnails_generated']} generated, {stats['errors']} errors)"
                )

            # Generate thumbnail path
            thumb_path = get_thumbnail_path(checksum, thumbnails_dir)

            # Check if source file exists
            if not source_path.exists():
                logger.warning(f"Source file not found: {source_path}")
                stats["errors"] += 1
                continue

            # Generate thumbnail
            try:
                if generate_thumbnail(source_path, thumb_path):
                    # Update database with thumbnail path
                    relative_path = str(thumb_path.relative_to(catalog_path))
                    conn.execute(
                        text(
                            "UPDATE images SET thumbnail_path = :thumb_path WHERE id = :id"
                        ),
                        {"thumb_path": relative_path, "id": video_id},
                    )
                    stats["thumbnails_generated"] += 1
                else:
                    logger.warning(f"Failed to generate thumbnail: {source_path}")
                    stats["errors"] += 1
            except Exception as e:
                logger.error(f"Error generating thumbnail for {source_path}: {e}")
                stats["errors"] += 1

        # Commit the transaction
        conn.commit()

    logger.info("\nMigration completed!")
    logger.info(f"  Thumbnails generated: {stats['thumbnails_generated']:,}")
    logger.info(f"  Errors: {stats['errors']:,}")

    return stats


def main():
    """Run the migration."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate thumbnails for videos")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be changed without generating thumbnails",
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help="Path to catalog directory (default: /app/catalogs/default)",
    )
    args = parser.parse_args()

    try:
        stats = run_migration(catalog_path=args.catalog_path, dry_run=args.dry_run)
        logger.info(f"\nMigration stats: {stats}")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
