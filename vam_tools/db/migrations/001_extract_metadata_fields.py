#!/usr/bin/env python3
"""
Migration 001: Extract metadata fields from nested EXIF to top-level fields.

This migration extracts commonly-used metadata fields from the nested
metadata->'exif' structure to top-level metadata fields for easier querying
and filtering.

Fields extracted:
- GPS: gps_latitude, gps_longitude, gps_altitude
- Camera: camera_make, camera_model, lens_model
- Settings: focal_length, aperture (f-number), iso, shutter_speed
- Other: orientation, flash, artist, copyright

Run with: python -m vam_tools.db.migrations.001_extract_metadata_fields
"""

import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sqlalchemy import text  # noqa: E402

from vam_tools.db.connection import engine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_migration(dry_run: bool = False) -> dict:
    """
    Extract metadata fields from EXIF to top-level metadata fields.

    Args:
        dry_run: If True, only report what would be changed without modifying data.

    Returns:
        Dictionary with migration statistics.
    """
    stats = {
        "total_images": 0,
        "gps_updated": 0,
        "camera_updated": 0,
        "settings_updated": 0,
        "other_updated": 0,
        "errors": 0,
    }

    with engine.connect() as conn:
        # Get total count
        result = conn.execute(text("SELECT COUNT(*) FROM images"))
        stats["total_images"] = result.scalar()
        logger.info(f"Total images in database: {stats['total_images']:,}")

        # Check current state
        result = conn.execute(
            text(
                """
            SELECT
                COUNT(*) FILTER (WHERE metadata->>'gps_latitude' IS NOT NULL) as has_gps,
                COUNT(*) FILTER (WHERE metadata->>'camera_make' IS NOT NULL) as has_camera,
                COUNT(*) FILTER (WHERE metadata->'exif'->>'Composite:GPSLatitude' IS NOT NULL) as exif_gps,
                COUNT(*) FILTER (WHERE metadata->'exif'->>'EXIF:Make' IS NOT NULL) as exif_camera
            FROM images
        """
            )
        )
        row = result.fetchone()
        logger.info("Current state:")
        logger.info(f"  - Images with top-level GPS: {row[0]:,}")
        logger.info(f"  - Images with top-level camera: {row[1]:,}")
        logger.info(f"  - Images with EXIF GPS: {row[2]:,}")
        logger.info(f"  - Images with EXIF camera: {row[3]:,}")

        if dry_run:
            logger.info("DRY RUN - no changes will be made")
            return stats

        # Update GPS coordinates (use Composite values which have proper signs)
        logger.info("Extracting GPS coordinates...")
        result = conn.execute(
            text(
                """
            UPDATE images
            SET metadata = metadata || jsonb_build_object(
                'gps_latitude', (metadata->'exif'->>'Composite:GPSLatitude')::float,
                'gps_longitude', (metadata->'exif'->>'Composite:GPSLongitude')::float,
                'gps_altitude', (metadata->'exif'->>'Composite:GPSAltitude')::float
            ),
            updated_at = NOW()
            WHERE metadata->'exif'->>'Composite:GPSLatitude' IS NOT NULL
              AND (metadata->>'gps_latitude' IS NULL
                   OR (metadata->>'gps_latitude')::float = 0)
        """
            )
        )
        stats["gps_updated"] = result.rowcount
        logger.info(f"  Updated GPS for {stats['gps_updated']:,} images")

        # Update camera info
        logger.info("Extracting camera information...")
        result = conn.execute(
            text(
                """
            UPDATE images
            SET metadata = metadata || jsonb_build_object(
                'camera_make', metadata->'exif'->>'EXIF:Make',
                'camera_model', metadata->'exif'->>'EXIF:Model',
                'lens_model', COALESCE(
                    metadata->'exif'->>'EXIF:LensModel',
                    metadata->'exif'->>'Composite:LensID'
                )
            ),
            updated_at = NOW()
            WHERE metadata->'exif'->>'EXIF:Make' IS NOT NULL
              AND metadata->>'camera_make' IS NULL
        """
            )
        )
        stats["camera_updated"] = result.rowcount
        logger.info(f"  Updated camera info for {stats['camera_updated']:,} images")

        # Update camera settings
        # Note: Some fields may have unusual formats (e.g., ISO "50 0 0")
        # We use regex to validate numeric values before casting
        logger.info("Extracting camera settings...")
        result = conn.execute(
            text(
                """
            UPDATE images
            SET metadata = metadata || jsonb_build_object(
                'focal_length',
                    CASE WHEN metadata->'exif'->>'EXIF:FocalLength' ~ '^[0-9.]+$'
                         THEN (metadata->'exif'->>'EXIF:FocalLength')::float
                         ELSE NULL END,
                'aperture',
                    CASE WHEN metadata->'exif'->>'EXIF:FNumber' ~ '^[0-9.]+$'
                         THEN (metadata->'exif'->>'EXIF:FNumber')::float
                         ELSE NULL END,
                'iso',
                    CASE WHEN metadata->'exif'->>'EXIF:ISO' ~ '^[0-9]+$'
                         THEN (metadata->'exif'->>'EXIF:ISO')::int
                         ELSE NULL END,
                'shutter_speed',
                    CASE WHEN metadata->'exif'->>'EXIF:ExposureTime' ~ '^[0-9.]+$'
                         THEN (metadata->'exif'->>'EXIF:ExposureTime')::float
                         ELSE NULL END
            ),
            updated_at = NOW()
            WHERE (metadata->'exif'->>'EXIF:FocalLength' IS NOT NULL
                   OR metadata->'exif'->>'EXIF:FNumber' IS NOT NULL
                   OR metadata->'exif'->>'EXIF:ISO' IS NOT NULL
                   OR metadata->'exif'->>'EXIF:ExposureTime' IS NOT NULL)
              AND metadata->>'focal_length' IS NULL
        """
            )
        )
        stats["settings_updated"] = result.rowcount
        logger.info(f"  Updated settings for {stats['settings_updated']:,} images")

        # Update other metadata (orientation, flash, artist, copyright)
        logger.info("Extracting other metadata...")
        result = conn.execute(
            text(
                """
            UPDATE images
            SET metadata = metadata || jsonb_build_object(
                'orientation',
                    CASE WHEN metadata->'exif'->>'EXIF:Orientation' ~ '^[0-9]+$'
                         THEN (metadata->'exif'->>'EXIF:Orientation')::int
                         ELSE NULL END,
                'flash',
                    CASE WHEN metadata->'exif'->>'EXIF:Flash' ~ '^[0-9]+$'
                         THEN (metadata->'exif'->>'EXIF:Flash')::int
                         ELSE NULL END,
                'artist', metadata->'exif'->>'EXIF:Artist',
                'copyright', metadata->'exif'->>'EXIF:Copyright'
            ),
            updated_at = NOW()
            WHERE (metadata->'exif'->>'EXIF:Orientation' IS NOT NULL
                   OR metadata->'exif'->>'EXIF:Flash' IS NOT NULL
                   OR metadata->'exif'->>'EXIF:Artist' IS NOT NULL
                   OR metadata->'exif'->>'EXIF:Copyright' IS NOT NULL)
              AND metadata->>'orientation' IS NULL
        """
            )
        )
        stats["other_updated"] = result.rowcount
        logger.info(f"  Updated other metadata for {stats['other_updated']:,} images")

        # Commit the transaction
        conn.commit()

        # Verify the update
        result = conn.execute(
            text(
                """
            SELECT
                COUNT(*) FILTER (WHERE metadata->>'gps_latitude' IS NOT NULL
                                   AND (metadata->>'gps_latitude')::float != 0) as has_gps,
                COUNT(*) FILTER (WHERE metadata->>'camera_make' IS NOT NULL) as has_camera,
                COUNT(*) FILTER (WHERE metadata->>'focal_length' IS NOT NULL) as has_settings
            FROM images
        """
            )
        )
        row = result.fetchone()
        logger.info("\nFinal state:")
        logger.info(f"  - Images with top-level GPS: {row[0]:,}")
        logger.info(f"  - Images with top-level camera: {row[1]:,}")
        logger.info(f"  - Images with top-level settings: {row[2]:,}")

    logger.info("\nMigration completed successfully!")
    return stats


def main():
    """Run the migration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract metadata fields from EXIF to top-level"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be changed without modifying data",
    )
    args = parser.parse_args()

    try:
        stats = run_migration(dry_run=args.dry_run)
        logger.info(f"\nMigration stats: {stats}")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
