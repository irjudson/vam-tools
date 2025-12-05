#!/usr/bin/env python3
"""
Migration 003: Add geohash columns and populate them for images with GPS data.

Geohashes enable efficient spatial queries by encoding lat/lon into hierarchical
string prefixes. We store three precision levels:
- geohash_4: ~39km (country/region view, zoom 5-8)
- geohash_6: ~1.2km (city/neighborhood view, zoom 9-12)
- geohash_8: ~40m (street/building view, zoom 13+)

Run with: python -m vam_tools.db.migrations.003_add_geohash_columns
"""

import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sqlalchemy import text  # noqa: E402

from vam_tools.db.connection import engine  # noqa: E402
from vam_tools.shared.geohash import encode  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_migration(dry_run: bool = False) -> dict:
    """
    Add geohash columns and populate them for images with GPS data.

    Args:
        dry_run: If True, only report what would be changed without modifying data.

    Returns:
        Dictionary with migration statistics.
    """
    stats = {
        "columns_added": 0,
        "indexes_added": 0,
        "images_with_gps": 0,
        "geohashes_populated": 0,
        "errors": 0,
    }

    with engine.connect() as conn:
        # Check if columns already exist
        result = conn.execute(
            text(
                """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'images'
            AND column_name IN ('geohash_4', 'geohash_6', 'geohash_8')
        """
            )
        )
        existing_columns = {row[0] for row in result}
        logger.info(f"Existing geohash columns: {existing_columns}")

        if dry_run:
            logger.info("DRY RUN - no changes will be made")

        # Add columns if they don't exist
        columns_to_add = [
            ("geohash_4", "VARCHAR(4)"),
            ("geohash_6", "VARCHAR(6)"),
            ("geohash_8", "VARCHAR(8)"),
        ]

        for col_name, col_type in columns_to_add:
            if col_name not in existing_columns:
                logger.info(f"Adding column: {col_name}")
                if not dry_run:
                    conn.execute(
                        text(
                            f"ALTER TABLE images ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                        )
                    )
                    conn.commit()
                stats["columns_added"] += 1

        # Add indexes
        indexes = [
            ("idx_images_geohash_4", "geohash_4"),
            ("idx_images_geohash_6", "geohash_6"),
            ("idx_images_geohash_8", "geohash_8"),
        ]

        for idx_name, col_name in indexes:
            # Check if index exists
            result = conn.execute(
                text(
                    """
                SELECT 1 FROM pg_indexes
                WHERE indexname = :idx_name
            """
                ),
                {"idx_name": idx_name},
            )
            if not result.fetchone():
                logger.info(f"Creating index: {idx_name}")
                if not dry_run:
                    conn.execute(
                        text(
                            f"""
                        CREATE INDEX IF NOT EXISTS {idx_name}
                        ON images(catalog_id, {col_name})
                        WHERE {col_name} IS NOT NULL
                    """
                        )
                    )
                    conn.commit()
                stats["indexes_added"] += 1

        # Count images with GPS data
        result = conn.execute(
            text(
                """
            SELECT COUNT(*) FROM images
            WHERE metadata->>'gps_latitude' IS NOT NULL
            AND (metadata->>'gps_latitude')::float != 0
        """
            )
        )
        stats["images_with_gps"] = result.scalar()
        logger.info(f"Images with GPS data: {stats['images_with_gps']:,}")

        # Count images needing geohash population
        result = conn.execute(
            text(
                """
            SELECT COUNT(*) FROM images
            WHERE metadata->>'gps_latitude' IS NOT NULL
            AND (metadata->>'gps_latitude')::float != 0
            AND geohash_4 IS NULL
        """
            )
        )
        images_to_update = result.scalar()
        logger.info(f"Images needing geohash population: {images_to_update:,}")

        if dry_run:
            stats["geohashes_populated"] = images_to_update
            return stats

        if images_to_update == 0:
            logger.info("All images with GPS already have geohashes!")
            return stats

        # Populate geohashes in batches
        batch_size = 1000
        processed = 0

        while True:
            # Fetch batch of images needing geohashes
            result = conn.execute(
                text(
                    """
                SELECT id,
                       (metadata->>'gps_latitude')::float as lat,
                       (metadata->>'gps_longitude')::float as lon
                FROM images
                WHERE metadata->>'gps_latitude' IS NOT NULL
                AND (metadata->>'gps_latitude')::float != 0
                AND geohash_4 IS NULL
                LIMIT :batch_size
            """
                ),
                {"batch_size": batch_size},
            )
            rows = result.fetchall()

            if not rows:
                break

            for row in rows:
                image_id, lat, lon = row

                if lat is None or lon is None:
                    continue

                # Validate coordinates before computing geohash
                if lat < -90 or lat > 90 or lon < -180 or lon > 180:
                    logger.warning(
                        f"Invalid coordinates for image {image_id}: lat={lat}, lon={lon}"
                    )
                    # Mark as processed with empty string to prevent re-processing
                    conn.execute(
                        text(
                            """
                        UPDATE images
                        SET geohash_4 = '', geohash_6 = '', geohash_8 = ''
                        WHERE id = :image_id
                    """
                        ),
                        {"image_id": image_id},
                    )
                    stats["errors"] += 1
                    continue

                try:
                    # Compute geohashes at different precisions
                    gh4 = encode(lat, lon, precision=4)
                    gh6 = encode(lat, lon, precision=6)
                    gh8 = encode(lat, lon, precision=8)

                    conn.execute(
                        text(
                            """
                        UPDATE images
                        SET geohash_4 = :gh4, geohash_6 = :gh6, geohash_8 = :gh8
                        WHERE id = :image_id
                    """
                        ),
                        {"image_id": image_id, "gh4": gh4, "gh6": gh6, "gh8": gh8},
                    )
                    stats["geohashes_populated"] += 1
                except Exception as e:
                    logger.warning(f"Error computing geohash for image {image_id}: {e}")
                    # Mark as processed with empty string to prevent re-processing
                    conn.execute(
                        text(
                            """
                        UPDATE images
                        SET geohash_4 = '', geohash_6 = '', geohash_8 = ''
                        WHERE id = :image_id
                    """
                        ),
                        {"image_id": image_id},
                    )
                    stats["errors"] += 1

            conn.commit()
            processed += len(rows)

            if processed % 5000 == 0:
                logger.info(
                    f"Progress: {processed:,}/{images_to_update:,} "
                    f"({processed / images_to_update * 100:.1f}%)"
                )

    logger.info("\nMigration completed!")
    logger.info(f"  Columns added: {stats['columns_added']}")
    logger.info(f"  Indexes added: {stats['indexes_added']}")
    logger.info(f"  Geohashes populated: {stats['geohashes_populated']:,}")
    logger.info(f"  Errors: {stats['errors']}")

    return stats


def main():
    """Run the migration."""
    import argparse

    parser = argparse.ArgumentParser(description="Add geohash columns to images table")
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
