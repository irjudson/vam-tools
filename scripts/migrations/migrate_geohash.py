#!/usr/bin/env python3
"""
Migrate existing GPS coordinates to geohash metadata field.

This script populates the metadata.geohash field for all images that have
GPS coordinates but don't yet have a geohash value.
"""

import logging
import sys

import pygeohash as pgh
from sqlalchemy import text

from vam_tools.db import CatalogDB as CatalogDatabase

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def migrate_geohash(catalog_id: str, batch_size: int = 1000):
    """
    Populate metadata.geohash for images with GPS coordinates.

    Args:
        catalog_id: UUID of the catalog to migrate
        batch_size: Number of images to process per batch
    """
    logger.info(f"Starting geohash migration for catalog {catalog_id}")

    with CatalogDatabase(catalog_id) as db:
        # Count images needing migration
        result = db.session.execute(
            text(
                """
                SELECT COUNT(*) FROM images
                WHERE metadata->>'gps_latitude' IS NOT NULL
                  AND metadata->>'gps_longitude' IS NOT NULL
                  AND (metadata->>'geohash' IS NULL OR metadata->>'geohash' = '')
            """
            )
        )
        total_images = result.scalar() or 0

        if total_images == 0:
            logger.info("No images need geohash migration")
            return

        logger.info(f"Found {total_images} images needing geohash")

        # Process in batches
        offset = 0
        updated = 0
        errors = 0

        while offset < total_images:
            # Fetch batch
            result = db.session.execute(
                text(
                    """
                    SELECT id, metadata->>'gps_latitude' as lat, metadata->>'gps_longitude' as lon
                    FROM images
                    WHERE metadata->>'gps_latitude' IS NOT NULL
                      AND metadata->>'gps_longitude' IS NOT NULL
                      AND (metadata->>'geohash' IS NULL OR metadata->>'geohash' = '')
                    LIMIT :limit OFFSET :offset
                """
                ),
                {"limit": batch_size, "offset": offset},
            )

            batch = result.fetchall()
            if not batch:
                break

            # Update each image in batch
            for row in batch:
                image_id, lat_str, lon_str = row

                try:
                    lat = float(lat_str)
                    lon = float(lon_str)

                    # Generate geohash (precision 7 = ~153m resolution)
                    geohash = pgh.encode(lat, lon, precision=7)

                    # Update metadata
                    db.session.execute(
                        text(
                            """
                            UPDATE images
                            SET metadata = metadata || jsonb_build_object('geohash', :geohash)
                            WHERE id = :image_id
                        """
                        ),
                        {"image_id": image_id, "geohash": geohash},
                    )
                    updated += 1

                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to generate geohash for image {image_id}: {e}"
                    )
                    errors += 1

            # Commit batch
            db.session.commit()

            offset += len(batch)
            logger.info(
                f"Progress: {offset}/{total_images} images processed "
                f"({updated} updated, {errors} errors)"
            )

        logger.info(
            f"Geohash migration complete: {updated} images updated, {errors} errors"
        )


if __name__ == "__main__":
    # Get catalog ID from environment or use default
    import os

    catalog_id = os.getenv("CATALOG_ID", "bd40ca52-c3f7-4877-9c97-1c227389c8c4")

    if len(sys.argv) > 1:
        catalog_id = sys.argv[1]

    migrate_geohash(catalog_id)
