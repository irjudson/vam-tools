"""
Migration script to convert JSON catalog to SQLite database.

This script reads the legacy catalog.json format and imports all data
into the new SQLite database schema.

Example:
    python -m vam_tools.core.migrate_to_sqlite /path/to/catalog
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from vam_tools.core.database import CatalogDatabase

logger = logging.getLogger(__name__)


class CatalogMigrator:
    """Migrates legacy JSON catalog to SQLite."""

    def __init__(self, catalog_path: Path) -> None:
        """Initialize migrator.

        Args:
            catalog_path: Path to catalog directory
        """
        self.catalog_path = Path(catalog_path)
        self.json_file = self.catalog_path / "catalog.json"
        self.db = CatalogDatabase(catalog_path)

    def migrate(self, dry_run: bool = False) -> Dict[str, int]:
        """Migrate JSON catalog to SQLite.

        Args:
            dry_run: If True, don't write to database, just report stats

        Returns:
            Dictionary with migration statistics

        Example:
            >>> migrator = CatalogMigrator(Path("catalog"))
            >>> stats = migrator.migrate(dry_run=True)
            >>> print(f"Would migrate {stats['images']} images")
        """
        if not self.json_file.exists():
            raise FileNotFoundError(f"Catalog not found: {self.json_file}")

        # Load JSON data
        logger.info(f"Loading JSON catalog from {self.json_file}")
        with open(self.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        stats = {
            "images": 0,
            "tags": 0,
            "image_tags": 0,
            "duplicate_groups": 0,
            "duplicate_images": 0,
            "burst_groups": 0,
            "burst_images": 0,
            "review_queue": 0,
            "problematic_files": 0,
        }

        if dry_run:
            # Count what would be migrated
            stats["images"] = len(data.get("images", {}))
            stats["duplicate_groups"] = len(data.get("duplicate_groups", {}))
            stats["burst_groups"] = len(data.get("burst_groups", {}))
            stats["review_queue"] = len(data.get("review_queue", []))
            stats["problematic_files"] = len(data.get("problematic_files", {}))

            # Count duplicate images
            for group in data.get("duplicate_groups", {}).values():
                stats["duplicate_images"] += len(group.get("images", []))

            # Count burst images
            for group in data.get("burst_groups", {}).values():
                stats["burst_images"] += len(group.get("images", []))

            logger.info(f"Dry run statistics: {stats}")
            return stats

        # Perform actual migration
        self.db.connect()
        self.db.initialize()

        with self.db.transaction():
            # Migrate catalog configuration
            self._migrate_config(data.get("configuration", {}))

            # Migrate images
            stats["images"] = self._migrate_images(data.get("images", {}))

            # Migrate duplicate groups
            dup_stats = self._migrate_duplicate_groups(
                data.get("duplicate_groups", {})
            )
            stats["duplicate_groups"] = dup_stats[0]
            stats["duplicate_images"] = dup_stats[1]

            # Migrate burst groups
            burst_stats = self._migrate_burst_groups(data.get("burst_groups", {}))
            stats["burst_groups"] = burst_stats[0]
            stats["burst_images"] = burst_stats[1]

            # Migrate review queue
            stats["review_queue"] = self._migrate_review_queue(
                data.get("review_queue", [])
            )

            # Migrate problematic files
            stats["problematic_files"] = self._migrate_problematic_files(
                data.get("problematic_files", {})
            )

            # Migrate statistics if present
            if "statistics" in data:
                self._migrate_statistics(data["statistics"])

        logger.info(f"Migration completed: {stats}")
        return stats

    def _migrate_config(self, config: Dict[str, Any]) -> None:
        """Migrate catalog configuration.

        Args:
            config: Configuration dictionary from JSON
        """
        for key, value in config.items():
            self.db.execute(
                "INSERT OR REPLACE INTO catalog_config "
                "(key, value, updated_at) VALUES (?, ?, datetime('now'))",
                (key, json.dumps(value)),
            )
        logger.debug("Migrated catalog configuration")

    def _migrate_images(self, images: Dict[str, Dict[str, Any]]) -> int:
        """Migrate image records.

        Args:
            images: Dictionary of image_id -> image_data

        Returns:
            Number of images migrated
        """
        count = 0
        for image_id, img_data in images.items():
            self.db.execute(
                """
                INSERT OR REPLACE INTO images (
                    id, source_path, organized_path, file_size, file_hash,
                    format, width, height, created_at, modified_at, indexed_at,
                    date_taken, camera_make, camera_model, lens_model,
                    focal_length, aperture, shutter_speed, iso,
                    gps_latitude, gps_longitude,
                    quality_score, is_corrupted, perceptual_hash
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    image_id,
                    img_data.get("source_path"),
                    img_data.get("organized_path"),
                    img_data.get("file_size"),
                    img_data.get("file_hash"),
                    img_data.get("format"),
                    img_data.get("width"),
                    img_data.get("height"),
                    img_data.get("created_at"),
                    img_data.get("modified_at"),
                    img_data.get("indexed_at"),
                    img_data.get("date_taken"),
                    img_data.get("camera_make"),
                    img_data.get("camera_model"),
                    img_data.get("lens_model"),
                    img_data.get("focal_length"),
                    img_data.get("aperture"),
                    img_data.get("shutter_speed"),
                    img_data.get("iso"),
                    img_data.get("gps_latitude"),
                    img_data.get("gps_longitude"),
                    img_data.get("quality_score"),
                    1 if img_data.get("is_corrupted") else 0,
                    img_data.get("perceptual_hash"),
                ),
            )
            count += 1

        logger.info(f"Migrated {count} images")
        return count

    def _migrate_duplicate_groups(
        self, groups: Dict[str, Dict[str, Any]]
    ) -> tuple[int, int]:
        """Migrate duplicate groups.

        Args:
            groups: Dictionary of group_id -> group_data

        Returns:
            Tuple of (groups_count, images_count)
        """
        groups_count = 0
        images_count = 0

        for group_id, group_data in groups.items():
            # Insert group
            cursor = self.db.execute(
                """
                INSERT INTO duplicate_groups (
                    hash_distance, similarity_score, created_at, reviewed
                ) VALUES (?, ?, datetime('now'), ?)
                """,
                (
                    group_data.get("hash_distance", 0),
                    group_data.get("similarity_score"),
                    1 if group_data.get("reviewed") else 0,
                ),
            )
            new_group_id = cursor.lastrowid
            groups_count += 1

            # Insert images in group
            for img_data in group_data.get("images", []):
                self.db.execute(
                    """
                    INSERT INTO duplicate_group_images (
                        group_id, image_id, is_primary, quality_score
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        new_group_id,
                        img_data.get("id"),
                        1 if img_data.get("is_primary") else 0,
                        img_data.get("quality_score"),
                    ),
                )
                images_count += 1

        logger.info(
            f"Migrated {groups_count} duplicate groups with {images_count} images"
        )
        return groups_count, images_count

    def _migrate_burst_groups(
        self, groups: Dict[str, Dict[str, Any]]
    ) -> tuple[int, int]:
        """Migrate burst groups.

        Args:
            groups: Dictionary of group_id -> group_data

        Returns:
            Tuple of (groups_count, images_count)
        """
        groups_count = 0
        images_count = 0

        for group_id, group_data in groups.items():
            # Insert group
            cursor = self.db.execute(
                """
                INSERT INTO burst_groups (
                    time_window_seconds, created_at
                ) VALUES (?, datetime('now'))
                """,
                (group_data.get("time_window_seconds"),),
            )
            new_group_id = cursor.lastrowid
            groups_count += 1

            # Insert images in group
            for img_data in group_data.get("images", []):
                self.db.execute(
                    """
                    INSERT INTO burst_group_images (
                        group_id, image_id, sequence_number, is_best
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        new_group_id,
                        img_data.get("id"),
                        img_data.get("sequence_number"),
                        1 if img_data.get("is_best") else 0,
                    ),
                )
                images_count += 1

        logger.info(f"Migrated {groups_count} burst groups with {images_count} images")
        return groups_count, images_count

    def _migrate_review_queue(self, queue: list[Dict[str, Any]]) -> int:
        """Migrate review queue.

        Args:
            queue: List of review items

        Returns:
            Number of items migrated
        """
        count = 0
        for item in queue:
            self.db.execute(
                """
                INSERT INTO review_queue (
                    image_id, reason, priority, created_at, reviewed_at, action
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item.get("image_id"),
                    item.get("reason"),
                    item.get("priority", 0),
                    item.get("created_at"),
                    item.get("reviewed_at"),
                    item.get("action"),
                ),
            )
            count += 1

        logger.info(f"Migrated {count} review queue items")
        return count

    def _migrate_problematic_files(self, files: Dict[str, Dict[str, Any]]) -> int:
        """Migrate problematic files.

        Args:
            files: Dictionary of file_id -> file_data

        Returns:
            Number of files migrated
        """
        count = 0
        for file_id, file_data in files.items():
            self.db.execute(
                """
                INSERT INTO problematic_files (
                    file_path, category, error_message, detected_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    file_data.get("file_path"),
                    file_data.get("category"),
                    file_data.get("error_message"),
                    file_data.get("detected_at"),
                    file_data.get("resolved_at"),
                ),
            )
            count += 1

        logger.info(f"Migrated {count} problematic files")
        return count

    def _migrate_statistics(self, stats: Dict[str, Any]) -> None:
        """Migrate statistics snapshot.

        Args:
            stats: Statistics dictionary
        """
        self.db.execute(
            """
            INSERT INTO statistics (
                timestamp, total_images, total_size_bytes,
                images_scanned, images_hashed, images_tagged,
                duplicate_groups, duplicate_images, potential_savings_bytes,
                high_quality_count, medium_quality_count, low_quality_count,
                corrupted_count, unsupported_count,
                processing_time_seconds, images_per_second
            ) VALUES (
                datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                stats.get("total_images", 0),
                stats.get("total_size_bytes", 0),
                stats.get("images_scanned", 0),
                stats.get("images_hashed", 0),
                stats.get("images_tagged", 0),
                stats.get("duplicate_groups", 0),
                stats.get("duplicate_images", 0),
                stats.get("potential_savings_bytes", 0),
                stats.get("high_quality_count", 0),
                stats.get("medium_quality_count", 0),
                stats.get("low_quality_count", 0),
                stats.get("corrupted_count", 0),
                stats.get("unsupported_count", 0),
                stats.get("processing_time_seconds", 0),
                stats.get("images_per_second", 0),
            ),
        )
        logger.debug("Migrated statistics")


def main():
    """CLI entry point for migration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate JSON catalog to SQLite database"
    )
    parser.add_argument("catalog_path", type=Path, help="Path to catalog directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without writing",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        migrator = CatalogMigrator(args.catalog_path)
        stats = migrator.migrate(dry_run=args.dry_run)

        print("\n=== Migration Statistics ===")
        for key, value in stats.items():
            print(f"{key:.<30} {value:>6}")

        if args.dry_run:
            print("\nDry run complete. Run without --dry-run to perform migration.")
        else:
            print("\nMigration complete!")
            print(f"SQLite database: {migrator.db.db_path}")

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
